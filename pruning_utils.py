import torch
import torch.nn.functional as F
import numpy as np

def get_filter_importance(weight, activation_stats=None, alpha=0.7):
    """
    Calculate filter importance using both weight magnitude and activation statistics
    
    Args:
        weight: Filter weights
        activation_stats: Statistics of filter activations (optional)
        alpha: Weight for L1-norm importance
        
    Returns:
        Importance scores for each filter
    """
    # Calculate L1-norm importance
    l1_norm = torch.sum(torch.abs(weight), dim=[1, 2, 3])
    
    if activation_stats is not None:
        # Combine with activation statistics
        importance = alpha * l1_norm + (1 - alpha) * activation_stats
        return importance
    else:
        return l1_norm

def distillation_loss(student_output, teacher_output, temperature=3.0):
    """
    Knowledge distillation loss function
    
    Args:
        student_output: Output logits from student model
        teacher_output: Output logits from teacher model
        temperature: Temperature for softening probability distributions
        
    Returns:
        Distillation loss
    """
    soft_log_probs = F.log_softmax(student_output / temperature, dim=1)
    soft_targets = F.softmax(teacher_output / temperature, dim=1)
    distillation = F.kl_div(soft_log_probs, soft_targets, reduction='batchmean') * (temperature ** 2)
    return distillation

def update_bn_stats(model, train_loader, device):
    """
    Update BatchNorm running statistics after pruning
    
    Args:
        model: The pruned model
        train_loader: Training data loader
        device: Device to use
    """
    model.train()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            model(inputs)
            # Process only a subset of batches to save time
            if batch_idx >= 100:
                break

def collect_activation_stats(model, data_loader, device):
    """
    Collect activation statistics for each layer
    
    Args:
        model: The model
        data_loader: Data loader
        device: Device to use
    """
    # Register hooks to collect activations
    activations = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            activations[name] = output.abs().mean([0, 2, 3]).detach()
        return hook
    
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            hooks.append(module.register_forward_hook(hook_fn(name)))
    
    # Collect activations
    model.eval()
    with torch.no_grad():
        for i, (inputs, _) in enumerate(data_loader):
            inputs = inputs.to(device)
            model(inputs)
            if i >= 10:  # Limit to 10 batches for efficiency
                break
    
    # Store activation statistics in the model
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) and name in activations:
            module.activation_stats = activations[name]
    
    # Remove hooks
    for hook in hooks:
        hook.remove()

def calculate_layer_sensitivity(model, data_loader, device, prune_rates=[0.1, 0.3, 0.5, 0.7, 0.9]):
    """
    Calculate sensitivity of each layer to pruning
    
    Args:
        model: The model
        data_loader: Data loader
        device: Device to use
        prune_rates: List of pruning rates to test
        
    Returns:
        Dictionary mapping layer indices to sensitivity scores
    """
    import copy
    
    # Get baseline accuracy
    baseline_acc = evaluate_model(model, data_loader, device)
    
    # Calculate sensitivity for each layer
    sensitivity = {}
    
    for layer_idx in range(1, 56):  # For ResNet-56
        layer_sensitivity = []
        
        for rate in prune_rates:
            # Create a copy of the model
            model_copy = copy.deepcopy(model)
            
            # Prune the layer
            mask = mask_model(model_copy, layer_idx, rate)
            model_copy = apply_mask(model_copy, layer_idx, mask)
            
            # Evaluate the pruned model
            pruned_acc = evaluate_model(model_copy, data_loader, device)
            
            # Calculate accuracy drop
            acc_drop = baseline_acc - pruned_acc
            layer_sensitivity.append(acc_drop)
        
        # Average sensitivity across pruning rates
        sensitivity[layer_idx] = sum(layer_sensitivity) / len(layer_sensitivity)
    
    return sensitivity

def evaluate_model(model, data_loader, device):
    """
    Evaluate model accuracy
    
    Args:
        model: The model
        data_loader: Data loader
        device: Device to use
        
    Returns:
        Accuracy
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return correct / total

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
    
    # Use the enhanced filter importance function
    if hasattr(layer, 'activation_stats'):
        importance = get_filter_importance(weight, layer.activation_stats)
    else:
        # Calculate L1-norm
        importance = get_filter_importance(weight)
    
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
