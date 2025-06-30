from __future__ import division
import os, sys, shutil, time, random
import argparse
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as transforms
from utils import AverageMeter, RecorderMeter, time_string, convert_secs2time
import models
import numpy as np
import math

# Added adaptive pruning functions
def get_adaptive_rate(layer_idx, base_rate=0.7):
    """
    Return adaptive pruning rate based on layer position
    Early layers and final layers are more sensitive, so prune less
    """
    if layer_idx <= 18:  # First stage
        return base_rate + 0.1  # Prune less (keep more filters)
    elif layer_idx >= 37:  # Last stage
        return base_rate + 0.05  # Prune slightly less
    else:  # Middle stage
        return base_rate  # Standard pruning rate

def get_pruning_rate(epoch, max_epochs, final_rate=0.7, warmup_epochs=5):
    """
    Calculate progressive pruning rate
    
    Args:
        epoch: Current epoch
        max_epochs: Total number of epochs
        final_rate: Target pruning rate
        warmup_epochs: Number of warmup epochs
        
    Returns:
        Current pruning rate
    """
    if epoch < warmup_epochs:
        # No pruning during warmup
        return 1.0
    else:
        # Gradually increase pruning (decrease keep rate)
        current_rate = 1.0 - (1.0 - final_rate) * min(1.0, (epoch - warmup_epochs) / (max_epochs * 0.6))
        return current_rate

def cosine_annealing_lr(optimizer, epoch, max_epochs, initial_lr):
    """
    Cosine annealing learning rate scheduler
    
    Args:
        optimizer: Optimizer to update
        epoch: Current epoch
        max_epochs: Total number of epochs
        initial_lr: Initial learning rate
        
    Returns:
        Updated optimizer
    """
    lr = initial_lr * 0.5 * (1 + math.cos(math.pi * epoch / max_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Trains ResNeXt on CIFAR or ImageNet',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('data_path', type=str, help='Path to dataset')
parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100', 'imagenet', 'svhn', 'stl10'],
                    help='Choose between Cifar10/100 and ImageNet.')
parser.add_argument('--arch', metavar='ARCH', default='resnet18', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnext29_8_64)')
# Optimization options
parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
parser.add_argument('--learning_rate', type=float, default=0.1, help='The Learning Rate.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', type=float, default=0.0005, help='Weight decay (L2 penalty).')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gammas', type=float, nargs='+', default=[0.1, 0.1],
                    help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
# Checkpoints
parser.add_argument('--print_freq', default=200, type=int, metavar='N', help='print frequency (default: 200)')
parser.add_argument('--save_path', type=str, default='./', help='Folder to save checkpoints and log.')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers (default: 2)')
# random seed
parser.add_argument('--manualSeed', type=int, help='manual seed')
# compress rate
parser.add_argument('--rate', type=float, default=0.9, help='compress rate of model')
parser.add_argument('--layer_begin', type=int, default=1, help='compress layer of model')
parser.add_argument('--layer_end', type=int, default=1, help='compress layer of model')
parser.add_argument('--layer_inter', type=int, default=1, help='compress layer of model')
parser.add_argument('--epoch_prune', type=int, default=1, help='compress layer of model')
parser.add_argument('--use_state_dict', dest='use_state_dict', action='store_true', help='use state dcit or not')

# Added optimization arguments
parser.add_argument('--use_adaptive_rate', action='store_true', help='Use adaptive pruning rates based on layer position')
parser.add_argument('--use_progressive_pruning', action='store_true', help='Use progressive pruning schedule')
parser.add_argument('--use_cosine_lr', action='store_true', help='Use cosine annealing learning rate scheduler')
parser.add_argument('--use_distillation', action='store_true', help='Use knowledge distillation during pruning')
parser.add_argument('--use_activation_stats', action='store_true', help='Use activation statistics for pruning')
parser.add_argument('--distill_temp', type=float, default=3.0, help='Temperature for knowledge distillation')
parser.add_argument('--warmup_epochs', type=int, default=5, help='Number of warmup epochs before pruning')

args = parser.parse_args()
args.use_cuda = args.ngpu > 0 and torch.cuda.is_available()

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if args.use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)
cudnn.benchmark = True


def main():
    # Init logger
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    log = open(os.path.join(args.save_path, 'log_seed_{}.txt'.format(args.manualSeed)), 'w')
    print_log('save path : {}'.format(args.save_path), log)
    state = {k: v for k, v in args._get_kwargs()}
    print_log(state, log)
    print_log("Random Seed: {}".format(args.manualSeed), log)
    print_log("python version : {}".format(sys.version.replace('\n', ' ')), log)
    print_log("torch version : {}".format(torch.__version__), log)
    print_log("cudnn version : {}".format(torch.backends.cudnn.version()), log)
    print_log("Compress Rate: {}".format(args.rate), log)
    print_log("Layer Begin: {}".format(args.layer_begin), log)
    print_log("Layer End: {}".format(args.layer_end), log)
    print_log("Layer Inter: {}".format(args.layer_inter), log)
    print_log("Epoch prune: {}".format(args.epoch_prune), log)
    
    # Print optimization settings
    if args.use_adaptive_rate:
        print_log("Using adaptive pruning rates", log)
    if args.use_progressive_pruning:
        print_log("Using progressive pruning schedule with warmup epochs: {}".format(args.warmup_epochs), log)
    if args.use_cosine_lr:
        print_log("Using cosine annealing learning rate scheduler", log)
    if args.use_distillation:
        print_log("Using knowledge distillation with temperature: {}".format(args.distill_temp), log)
    if args.use_activation_stats:
        print_log("Using activation statistics for pruning", log)

    # Init dataset
    if not os.path.isdir(args.data_path):
        os.makedirs(args.data_path)

    if args.dataset == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif args.dataset == 'cifar100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
    else:
        assert False, "Unknow dataset : {}".format(args.dataset)

    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(),
         transforms.Normalize(mean, std)])
    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)])

    if args.dataset == 'cifar10':
        train_data = dset.CIFAR10(args.data_path, train=True, transform=train_transform, download=True)
        test_data = dset.CIFAR10(args.data_path, train=False, transform=test_transform, download=True)
        num_classes = 10
    elif args.dataset == 'cifar100':
        train_data = dset.CIFAR100(args.data_path, train=True, transform=train_transform, download=True)
        test_data = dset.CIFAR100(args.data_path, train=False, transform=test_transform, download=True)
        num_classes = 100
    elif args.dataset == 'svhn':
        train_data = dset.SVHN(args.data_path, split='train', transform=train_transform, download=True)
        test_data = dset.SVHN(args.data_path, split='test', transform=test_transform, download=True)
        num_classes = 10
    elif args.dataset == 'stl10':
        train_data = dset.STL10(args.data_path, split='train', transform=train_transform, download=True)
        test_data = dset.STL10(args.data_path, split='test', transform=test_transform, download=True)
        num_classes = 10
    elif args.dataset == 'imagenet':
        assert False, 'Do not finish imagenet code'
    else:
        assert False, 'Do not support dataset : {}'.format(args.dataset)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers, pin_memory=True)

    print_log("=> creating model '{}'".format(args.arch), log)
    # Init model, criterion, and optimizer
    net = models.__dict__[args.arch](num_classes)
    print_log("=> network :\n {}".format(net), log)

    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), state['learning_rate'], momentum=state['momentum'],
                                weight_decay=state['decay'], nesterov=True)

    if args.use_cuda:
        net.cuda()
        criterion.cuda()

    recorder = RecorderMeter(args.epochs)
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print_log("=> loading checkpoint '{}'".format(args.resume), log)
            checkpoint = torch.load(args.resume)
            recorder = checkpoint['recorder']
            args.start_epoch = checkpoint['epoch']
            if args.use_state_dict:
                net.load_state_dict(checkpoint['state_dict'])
            else:
                net = checkpoint['state_dict']

            optimizer.load_state_dict(checkpoint['optimizer'])
            print_log("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']), log)
        else:
            print_log("=> no checkpoint found at '{}'".format(args.resume), log)
    else:
        print_log("=> do not use any checkpoint for {} model".format(args.arch), log)

    if args.evaluate:
        time1 = time.time()
        validate(test_loader, net, criterion, log)
        time2 = time.time()
        print('function took %0.3f ms' % ((time2 - time1) * 1000.0))
        return

    # Create a copy of the original model for distillation if needed
    if args.use_distillation:
        import copy
        teacher_model = copy.deepcopy(net)
        teacher_model.eval()
        if args.use_cuda:
            teacher_model.cuda()
        print_log("Created teacher model for distillation", log)

    # Collect activation statistics if needed
    if args.use_activation_stats:
        from pruning_utils import collect_activation_stats
        print_log("Collecting activation statistics...", log)
        collect_activation_stats(net, train_loader, 'cuda' if args.use_cuda else 'cpu')

    # Main loop
    start_time = time.time()
    epoch_time = AverageMeter()
    for epoch in range(args.start_epoch, args.epochs):
        current_learning_rate = adjust_learning_rate(optimizer, epoch, args.gammas, args.schedule)
        
        # Use cosine annealing learning rate if enabled
        if args.use_cosine_lr:
            optimizer = cosine_annealing_lr(optimizer, epoch, args.epochs, args.learning_rate)
            current_learning_rate = optimizer.param_groups[0]['lr']

        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs - epoch))
        print_log(
            '\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [learning_rate={:6.4f}]'.format(time_string(), epoch, args.epochs,
                                                                                   convert_secs2time(
                                                                                       epoch_time.avg * epoch),
                                                                                   current_learning_rate) \
            + ' [Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs), log)

        # Train and test
        train_acc, train_los = train(train_loader, net, criterion, optimizer, epoch, log, teacher_model if args.use_distillation else None)
        test_acc, test_los = validate(test_loader, net, criterion, log)

        # Update recorder
        recorder.update(epoch, train_los, train_acc, test_los, test_acc)

        # Save model
        is_best = recorder.max_accuracy(False) == test_acc
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': net,
            'recorder': recorder,
            'optimizer': optimizer.state_dict(),
        }, is_best, args.save_path, 'checkpoint.pth.tar')

        # Measure elapsed time
        epoch_time.update(time.time() - start_time)
        start_time = time.time()
        recorder.plot_curve(os.path.join(args.save_path, 'curve.png'))

    log.close()


# train function
def train(train_loader, model, criterion, optimizer, epoch, log, teacher_model=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # Switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # Measure data loading time
        data_time.update(time.time() - end)

        if args.use_cuda:
            target = target.cuda()
            input = input.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # Compute output
        output = model(input_var)
        
        # Calculate loss (with distillation if enabled)
        if teacher_model is not None:
            # Get teacher predictions
            with torch.no_grad():
                teacher_output = teacher_model(input_var)
            
            # Regular cross-entropy loss
            ce_loss = criterion(output, target_var)
            
            # Distillation loss
            from pruning_utils import distillation_loss
            kd_loss = distillation_loss(output, teacher_output, args.distill_temp)
            
            # Combined loss
            alpha = 0.5  # Balance between distillation and regular loss
            loss = alpha * ce_loss + (1 - alpha) * kd_loss
        else:
            # Regular loss
            loss = criterion(output, target_var)

        # Measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # Compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print_log('  Epoch: [{:03d}][{:03d}/{:03d}]   '
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})   '
                      'Loss {loss.val:.4f} ({loss.avg:.4f})   '
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})   '.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5) + time_string(), log)
    
    print_log('  **Train** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(
        top1=top1, top5=top5, error1=100-top1.avg), log)
    
    # Apply pruning if needed
    if epoch % args.epoch_prune == 0:
        print_log('Pruning filters for epoch {}'.format(epoch), log)
        
        # Apply adaptive pruning rate if enabled
        if args.use_adaptive_rate:
            for layer_idx in range(args.layer_begin, args.layer_end + 1, args.layer_inter):
                adaptive_rate = get_adaptive_rate(layer_idx, args.rate)
                
                # Apply progressive pruning if enabled
                if args.use_progressive_pruning:
                    current_rate = get_pruning_rate(epoch, args.epochs, adaptive_rate, args.warmup_epochs)
                else:
                    current_rate = adaptive_rate
                
                print_log('Layer index: {:d}, pruning rate: {:.2f}'.format(layer_idx, current_rate), log)
                mask = mask_model(model, layer_idx, current_rate)
                model = apply_mask(model, layer_idx, mask)
        else:
            # Original pruning logic
            for layer_idx in range(args.layer_begin, args.layer_end + 1, args.layer_inter):
                mask = mask_model(model, layer_idx, args.rate)
                model = apply_mask(model, layer_idx, mask)
    
    # Update batch normalization statistics after pruning if needed
    if epoch % args.epoch_prune == 0 and epoch > 0:
        from pruning_utils import update_bn_stats
        print_log('Updating batch normalization statistics...', log)
        update_bn_stats(model, train_loader, 'cuda' if args.use_cuda else 'cpu')
    
    return top1.avg, losses.avg


def validate(val_loader, model, criterion, log):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # Switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if args.use_cuda:
                target = target.cuda()
                input = input.cuda()
            
            # Compute output
            output = model(input)
            loss = criterion(output, target)

            # Measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.data.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

    print_log('  **Test** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(
        top1=top1, top5=top5, error1=100-top1.avg), log)

    return top1.avg, losses.avg


def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()


def save_checkpoint(state, is_best, save_path, filename):
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)
    if is_best:
        bestname = os.path.join(save_path, 'model_best.pth.tar')
        shutil.copyfile(filename, bestname)


def adjust_learning_rate(optimizer, epoch, gammas, schedule):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.learning_rate
    assert len(gammas) == len(schedule), "length of gammas and schedule should be equal"
    for (gamma, step) in zip(gammas, schedule):
        if (epoch >= step):
            lr = lr * gamma
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def mask_model(model, layer_idx, prune_ratio):
    """
    Generate mask for pruning
    
    Args:
        model: The model
        layer_idx: Layer index
        prune_ratio: Pruning ratio
        
    Returns:
        Mask for pruning
    """
    # Find the layer
    layer = None
    if layer_idx == 1:
        layer = model.module.conv_1_3x3
    elif layer_idx == 2:
        layer = model.module.stage_1[0].conv_a
    elif layer_idx == 3:
        layer = model.module.stage_1[0].conv_b
    elif layer_idx == 4:
        layer = model.module.stage_1[1].conv_a
    elif layer_idx == 5:
        layer = model.module.stage_1[1].conv_b
    elif layer_idx == 6:
        layer = model.module.stage_1[2].conv_a
    elif layer_idx == 7:
        layer = model.module.stage_1[2].conv_b
    elif layer_idx == 8:
        layer = model.module.stage_1[3].conv_a
    elif layer_idx == 9:
        layer = model.module.stage_1[3].conv_b
    elif layer_idx == 10:
        layer = model.module.stage_1[4].conv_a
    elif layer_idx == 11:
        layer = model.module.stage_1[4].conv_b
    elif layer_idx == 12:
        layer = model.module.stage_1[5].conv_a
    elif layer_idx == 13:
        layer = model.module.stage_1[5].conv_b
    elif layer_idx == 14:
        layer = model.module.stage_1[6].conv_a
    elif layer_idx == 15:
        layer = model.module.stage_1[6].conv_b
    elif layer_idx == 16:
        layer = model.module.stage_1[7].conv_a
    elif layer_idx == 17:
        layer = model.module.stage_1[7].conv_b
    elif layer_idx == 18:
        layer = model.module.stage_1[8].conv_a
    elif layer_idx == 19:
        layer = model.module.stage_1[8].conv_b
    elif layer_idx == 20:
        layer = model.module.stage_2[0].conv_a
    elif layer_idx == 21:
        layer = model.module.stage_2[0].conv_b
    elif layer_idx == 22:
        layer = model.module.stage_2[1].conv_a
    elif layer_idx == 23:
        layer = model.module.stage_2[1].conv_b
    elif layer_idx == 24:
        layer = model.module.stage_2[2].conv_a
    elif layer_idx == 25:
        layer = model.module.stage_2[2].conv_b
    elif layer_idx == 26:
        layer = model.module.stage_2[3].conv_a
    elif layer_idx == 27:
        layer = model.module.stage_2[3].conv_b
    elif layer_idx == 28:
        layer = model.module.stage_2[4].conv_a
    elif layer_idx == 29:
        layer = model.module.stage_2[4].conv_b
    elif layer_idx == 30:
        layer = model.module.stage_2[5].conv_a
    elif layer_idx == 31:
        layer = model.module.stage_2[5].conv_b
    elif layer_idx == 32:
        layer = model.module.stage_2[6].conv_a
    elif layer_idx == 33:
        layer = model.module.stage_2[6].conv_b
    elif layer_idx == 34:
        layer = model.module.stage_2[7].conv_a
    elif layer_idx == 35:
        layer = model.module.stage_2[7].conv_b
    elif layer_idx == 36:
        layer = model.module.stage_2[8].conv_a
    elif layer_idx == 37:
        layer = model.module.stage_2[8].conv_b
    elif layer_idx == 38:
        layer = model.module.stage_3[0].conv_a
    elif layer_idx == 39:
        layer = model.module.stage_3[0].conv_b
    elif layer_idx == 40:
        layer = model.module.stage_3[1].conv_a
    elif layer_idx == 41:
        layer = model.module.stage_3[1].conv_b
    elif layer_idx == 42:
        layer = model.module.stage_3[2].conv_a
    elif layer_idx == 43:
        layer = model.module.stage_3[2].conv_b
    elif layer_idx == 44:
        layer = model.module.stage_3[3].conv_a
    elif layer_idx == 45:
        layer = model.module.stage_3[3].conv_b
    elif layer_idx == 46:
        layer = model.module.stage_3[4].conv_a
    elif layer_idx == 47:
        layer = model.module.stage_3[4].conv_b
    elif layer_idx == 48:
        layer = model.module.stage_3[5].conv_a
    elif layer_idx == 49:
        layer = model.module.stage_3[5].conv_b
    elif layer_idx == 50:
        layer = model.module.stage_3[6].conv_a
    elif layer_idx == 51:
        layer = model.module.stage_3[6].conv_b
    elif layer_idx == 52:
        layer = model.module.stage_3[7].conv_a
    elif layer_idx == 53:
        layer = model.module.stage_3[7].conv_b
    elif layer_idx == 54:
        layer = model.module.stage_3[8].conv_a
    elif layer_idx == 55:
        layer = model.module.stage_3[8].conv_b
    
    if layer is None:
        return None
    
    # Get the weights
    weight = layer.weight.data
    
    # Use the enhanced filter importance function if available
    if args.use_activation_stats and hasattr(layer, 'activation_stats'):
        from pruning_utils import get_filter_importance
        importance = get_filter_importance(weight, layer.activation_stats)
    else:
        # Calculate L1-norm
        importance = torch.sum(torch.abs(weight), dim=(1, 2, 3))
    
    # Sort and get threshold
    sorted_importance, sorted_idx = torch.sort(importance)
    threshold = sorted_importance[int(prune_ratio * len(sorted_importance))]
    
    # Create mask
    mask = torch.gt(importance, threshold).float()
    
    return mask


def apply_mask(model, layer_idx, mask):
    """
    Apply mask to the model
    
    Args:
        model: The model
        layer_idx: Layer index
        mask: Mask for pruning
        
    Returns:
        Pruned model
    """
    if mask is None:
        return model
    
    # Find the layer
    layer = None
    if layer_idx == 1:
        layer = model.module.conv_1_3x3
    elif layer_idx == 2:
        layer = model.module.stage_1[0].conv_a
    elif layer_idx == 3:
        layer = model.module.stage_1[0].conv_b
    elif layer_idx == 4:
        layer = model.module.stage_1[1].conv_a
    elif layer_idx == 5:
        layer = model.module.stage_1[1].conv_b
    elif layer_idx == 6:
        layer = model.module.stage_1[2].conv_a
    elif layer_idx == 7:
        layer = model.module.stage_1[2].conv_b
    elif layer_idx == 8:
        layer = model.module.stage_1[3].conv_a
    elif layer_idx == 9:
        layer = model.module.stage_1[3].conv_b
    elif layer_idx == 10:
        layer = model.module.stage_1[4].conv_a
    elif layer_idx == 11:
        layer = model.module.stage_1[4].conv_b
    elif layer_idx == 12:
        layer = model.module.stage_1[5].conv_a
    elif layer_idx == 13:
        layer = model.module.stage_1[5].conv_b
    elif layer_idx == 14:
        layer = model.module.stage_1[6].conv_a
    elif layer_idx == 15:
        layer = model.module.stage_1[6].conv_b
    elif layer_idx == 16:
        layer = model.module.stage_1[7].conv_a
    elif layer_idx == 17:
        layer = model.module.stage_1[7].conv_b
    elif layer_idx == 18:
        layer = model.module.stage_1[8].conv_a
    elif layer_idx == 19:
        layer = model.module.stage_1[8].conv_b
    elif layer_idx == 20:
        layer = model.module.stage_2[0].conv_a
    elif layer_idx == 21:
        layer = model.module.stage_2[0].conv_b
    elif layer_idx == 22:
        layer = model.module.stage_2[1].conv_a
    elif layer_idx == 23:
        layer = model.module.stage_2[1].conv_b
    elif layer_idx == 24:
        layer = model.module.stage_2[2].conv_a
    elif layer_idx == 25:
        layer = model.module.stage_2[2].conv_b
    elif layer_idx == 26:
        layer = model.module.stage_2[3].conv_a
    elif layer_idx == 27:
        layer = model.module.stage_2[3].conv_b
    elif layer_idx == 28:
        layer = model.module.stage_2[4].conv_a
    elif layer_idx == 29:
        layer = model.module.stage_2[4].conv_b
    elif layer_idx == 30:
        layer = model.module.stage_2[5].conv_a
    elif layer_idx == 31:
        layer = model.module.stage_2[5].conv_b
    elif layer_idx == 32:
        layer = model.module.stage_2[6].conv_a
    elif layer_idx == 33:
        layer = model.module.stage_2[6].conv_b
    elif layer_idx == 34:
        layer = model.module.stage_2[7].conv_a
    elif layer_idx == 35:
        layer = model.module.stage_2[7].conv_b
    elif layer_idx == 36:
        layer = model.module.stage_2[8].conv_a
    elif layer_idx == 37:
        layer = model.module.stage_2[8].conv_b
    elif layer_idx == 38:
        layer = model.module.stage_3[0].conv_a
    elif layer_idx == 39:
        layer = model.module.stage_3[0].conv_b
    elif layer_idx == 40:
        layer = model.module.stage_3[1].conv_a
    elif layer_idx == 41:
        layer = model.module.stage_3[1].conv_b
    elif layer_idx == 42:
        layer = model.module.stage_3[2].conv_a
    elif layer_idx == 43:
        layer = model.module.stage_3[2].conv_b
    elif layer_idx == 44:
        layer = model.module.stage_3[3].conv_a
    elif layer_idx == 45:
        layer = model.module.stage_3[3].conv_b
    elif layer_idx == 46:
        layer = model.module.stage_3[4].conv_a
    elif layer_idx == 47:
        layer = model.module.stage_3[4].conv_b
    elif layer_idx == 48:
        layer = model.module.stage_3[5].conv_a
    elif layer_idx == 49:
        layer = model.module.stage_3[5].conv_b
    elif layer_idx == 50:
        layer = model.module.stage_3[6].conv_a
    elif layer_idx == 51:
        layer = model.module.stage_3[6].conv_b
    elif layer_idx == 52:
        layer = model.module.stage_3[7].conv_a
    elif layer_idx == 53:
        layer = model.module.stage_3[7].conv_b
    elif layer_idx == 54:
        layer = model.module.stage_3[8].conv_a
    elif layer_idx == 55:
        layer = model.module.stage_3[8].conv_b
    
    if layer is None:
        return model
    
    # Apply mask
    layer.weight.data = layer.weight.data * mask.view(-1, 1, 1, 1)
    
    return model


if __name__ == '__main__':
    main()
