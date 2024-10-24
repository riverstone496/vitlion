import torch
from torch.distributed import all_reduce, ReduceOp


def sync_exp_avg(optimizer):
    """
    Synchronize the exp_avg (exponential moving averages) in the optimizer's state across all GPUs.
    
    Args:
        optimizer (Optimizer): The optimizer whose exp_avg state needs to be synced across GPUs.
    """
    # Ensure distributed training is initialized
    if not torch.distributed.is_initialized():
        return  # No-op if not in a distributed environment
    
    # Loop through the parameter groups and synchronize exp_avg for each parameter
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.grad is not None:
                state = optimizer.state[p]
                # Initialize exp_avg if not already present in the optimizer state
                if 'exp_avg' not in state:
                    state['exp_avg'] = torch.zeros_like(p)
                exp_avg = state['exp_avg']
                # Perform all_reduce to sum exp_avg across all GPUs
                all_reduce(exp_avg, op=ReduceOp.SUM)
                # Divide by the world size to get the average exp_avg across all GPUs
                world_size = torch.distributed.get_world_size()
                exp_avg.div_(world_size)

def sync_exp_avg_variance(optimizer, module_param_map, not_replace=False):
    """
    Compute the variance of the exp_avg (exponential moving averages) across GPUs.
    
    Args:
        optimizer (Optimizer): The optimizer whose exp_avg state needs to be synchronized and analyzed.
        hist_path (str, optional): Path to save histogram data. Defaults to None.
    """
    # Ensure distributed training is initialized
    logs = {}
    max_norm = 0
    min_norm = 1e+10
    mean_norm = 0
    param_size = 0
    if not torch.distributed.is_initialized():
        return  # No-op if not in a distributed environment
    # Loop through the parameter groups and calculate the mean of squares and variance
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.grad is not None:
                state = optimizer.state[p]
                # Initialize exp_avg if not already present in the optimizer state
                if 'exp_avg' not in state:
                    state['exp_avg'] = torch.zeros_like(p)
                exp_avg = state['exp_avg']
                # Calculate the mean of the squared elements of exp_avg on each worker
                local_mean_square = exp_avg ** 2
                # Perform all_reduce to get the sum of mean squares across all GPUs
                torch.distributed.all_reduce(local_mean_square, op=torch.distributed.ReduceOp.SUM)
                # Divide by the world size to get the global mean square
                world_size = torch.distributed.get_world_size()
                global_mean_square = local_mean_square / world_size
                
                if not_replace:
                    # Calculate the global mean of exp_avg
                    tmp_exp_avg = exp_avg.clone()
                    torch.distributed.all_reduce(tmp_exp_avg, op=torch.distributed.ReduceOp.SUM)
                    tmp_exp_avg.div_(world_size)
                    # Compute the variance as E[X^2] - (E[X])^2
                    variance = global_mean_square - tmp_exp_avg ** 2
                else:
                    # Calculate the global mean of exp_avg
                    torch.distributed.all_reduce(exp_avg, op=torch.distributed.ReduceOp.SUM)
                    exp_avg.div_(world_size)
                    # Compute the variance as E[X^2] - (E[X])^2
                    variance = global_mean_square - exp_avg ** 2
                module_name = module_param_map.get(p, "Unknown")
                logs[f"variance_l2/{module_name}"] = torch.norm(variance)
                logs[f"variance_mean/{module_name}"] = torch.mean(variance)
                logs[f"variance_max/{module_name}"] = torch.max(variance)
                logs[f"variance_min/{module_name}"] = torch.min(variance)
                if max_norm  < torch.max(variance):
                    max_norm  = torch.max(variance)
                if min_norm  > torch.min(variance):
                    min_norm  = torch.min(variance)
                mean_norm += torch.sum(variance)
                param_size += torch.numel(variance)
    logs[f"variance_max/all"] = max_norm
    logs[f"variance_min/all"] = min_norm
    logs[f"variance_mean/all"] = mean_norm / param_size
    return logs

def calculate_Tv(iter, k=16):
    if k < 0:
        return []
    Tv = [0]  # Start with the first variance update at step 0
    j = 0
    while True:
        next_step = Tv[-1] + 2 ** (j / k)
        if next_step > iter:
            break
        Tv.append(int(next_step))
        j += 1
    return Tv