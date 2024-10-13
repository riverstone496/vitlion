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