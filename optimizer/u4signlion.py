from typing import Callable, Optional, Tuple

import torch
from torch.optim.optimizer import Optimizer
from torch.distributed import all_reduce, ReduceOp
import time, os
from torch.nn.utils import parameters_to_vector, vector_to_parameters

# functions
def exists(val):
    return val is not None

def compress_uint8_to_uint4(x: torch.Tensor) -> torch.Tensor:
    """
    Compresses a tensor of uint8 values into uint4 pairs. Each uint8 tensor element is
    assumed to be in the range of uint4 (0 to 15). The result is a tensor that is half the size.

    Args:
        x (torch.Tensor): Input uint8 tensor with values in the range [0, 15].

    Returns:
        torch.Tensor: Compressed uint8 tensor, where each element contains two uint4 values.
    """
    # Prepare the tensor for compression
    x = x.view(-1)  # Flatten if necessary

    # Split the elements into high (left 4 bits) and low (right 4 bits)
    high = (x[0::2] & 0x0F) << 4  # Get the high bits and shift them to the left 4 bits
    low = x[1::2] & 0x0F          # Get the low bits

    # Combine the high and low parts into one uint8 element
    compressed = (high | low).to(torch.uint8)

    return compressed

def decompress_uint4_to_uint8(compressed: torch.Tensor) -> torch.Tensor:
    """
    Decompresses a tensor of uint8 values where each element contains two uint4 values,
    back to the original uint8 tensor.

    Args:
        compressed (torch.Tensor): Compressed uint8 tensor with two uint4 values per element.

    Returns:
        torch.Tensor: Decompressed uint8 tensor with values in the range [0, 15].
    """

    # Extract the high and low 4-bit parts
    high = (compressed >> 4) & 0x0F  # Extract the high 4 bits
    low = compressed & 0x0F          # Extract the low 4 bits

    # Interleave the high and low parts to restore the original tensor
    decompressed = torch.empty(compressed.numel() * 2, dtype=torch.uint8, device=compressed.device)
    decompressed[0::2] = high
    decompressed[1::2] = low

    return decompressed

def compress_allreduce(x1, ngpus):
    x1t = torch.where(x1 == -1, torch.tensor(0, dtype=torch.uint8, device=x1.device), torch.tensor(1, dtype=torch.uint8, device=x1.device))
    torch.cuda.synchronize()  
    x1_compressed = compress_uint8_to_uint4(x1t)
    all_reduce(x1_compressed)
    torch.cuda.synchronize()  
    x1_recovered = decompress_uint4_to_uint8(x1_compressed).to(torch.int8)
    x1_recovered -= ngpus //2
    x1 = x1_recovered.sign()
    return x1

# update functions
def compute_updates_sign_int8(params, grads, exp_avgs, lr, wd, beta1):
    """
    Compute the sign-based updates for all parameters and aggregate them into a single vector.
    """
    updates = []
    for p, grad, exp_avg in zip(params, grads, exp_avgs):
        # Step weight decay
        p.data.mul_(1 - lr * wd)
        # Compute update: update = sign(beta1 * exp_avg + (1 - beta1) * grad)
        update = exp_avg.mul(beta1).add(grad, alpha=1 - beta1).sign().to(torch.int8)
        updates.append(update.flatten())
    # Concatenate all updates into a single vector
    update_vector = torch.cat(updates)
    return update_vector

# class
class U4SignLion(Optimizer):
    def __init__(
        self, 
        params, 
        lr: float = 1e-4, 
        betas: Tuple[float, float] = (0.9, 0.99), 
        weight_decay: float = 0.0, 
        use_triton: bool = False
    ):
        assert lr > 0.0, "Learning rate must be positive."
        assert all([0.0 <= beta <= 1.0 for beta in betas]), "Betas must be between 0 and 1."
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.beta1, self.beta2 = betas
        if use_triton:
            from lion_pytorch.triton import update_fn as triton_update_fn
            self.update_fn = triton_update_fn
        else:
            self.update_fn = None  # Not used in optimized step
        self.ngpus = int(os.getenv('OMPI_COMM_WORLD_SIZE', '1'))

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if exists(closure):
            with torch.enable_grad():
                loss = closure()
        
        params = []
        grads = []
        exp_avgs = []
        lr = self.defaults['lr']
        wd = self.defaults['weight_decay']
        beta1 = self.beta1
        beta2 = self.beta2
        original_dtype = self.param_groups[0]['params'][0].grad.dtype

        # Collect parameters, gradients, and exponential averages
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    params.append(p)
                    grads.append(p.grad)
                    state = self.state[p]
                    if 'exp_avg' not in state:
                        state['exp_avg'] = torch.zeros_like(p)
                    exp_avgs.append(state['exp_avg'])

        if not params:
            return loss  # No parameters to update

        # Compute updates for all parameters and aggregate into a single vector
        update_vector = compute_updates_sign_int8(params, grads, exp_avgs, lr, wd, beta1)

        # Perform all_reduce on the aggregated update vector
        if torch.distributed.is_initialized():
            torch.cuda.synchronize()  # Ensure all previous operations are complete
            start_time = time.time()
            update_vector = compress_allreduce(update_vector, ngpus=self.ngpus)
            torch.cuda.synchronize()  # Ensure all_reduce is complete
            elapsed_time = time.time() - start_time
            # Normalize the update if necessary (e.g., divide by world size)
            update_vector = update_vector.sign_()
        else:
            elapsed_time = 0.0

        # Apply the updates back to each parameter
        idx = 0  # Pointer to traverse the update_vector
        for p, exp_avg in zip(params, exp_avgs):
            numel = p.numel()
            # Extract the corresponding update for this parameter
            update = update_vector[idx:idx + numel].view_as(p).to(original_dtype)
            idx += numel
            # Apply the update
            p.add_(update, alpha=-lr)
            # Update the exponential moving average
            exp_avg.mul_(beta2).add_(p.grad, alpha=1 - beta2)

        self.elapsed_time = elapsed_time
        self.numel = idx
        return loss

# from typing import Callable, Optional, Tuple

# import torch
# from torch.optim.optimizer import Optimizer
# from torch.distributed import all_reduce, ReduceOp
# import time

# # functions
# def exists(val):
#     return val is not None

# # update functions
# def update_fn_sign(p, grad, exp_avg, lr, wd, beta1, beta2):
#     # stepweight decay
#     p.data.mul_(1 - lr * wd)
#     # weight update
#     update = exp_avg.clone().mul_(beta1).add(grad, alpha=1 - beta1).sign_()
#     if torch.distributed.is_initialized():
#         update = update.to(torch.int8)
#         # Time measurement for all_reduce
#         torch.cuda.synchronize()  # Ensure previous operations are complete
#         start_time = time.time()
#         all_reduce(update, op=ReduceOp.SUM)
#         torch.cuda.synchronize()  # Ensure all_reduce is complete
#         elapsed_time = time.time() - start_time
#         update = update.sign()
#     p.add_(update, alpha=-lr)
#     # decay the momentum running average coefficient
#     exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
#     return elapsed_time

# # class
# class SignLion(Optimizer):
#     def __init__(
#         self, params, lr: float = 1e-4, betas: Tuple[float, float] = (0.9, 0.99), weight_decay: float = 0.0, use_triton: bool = False
#     ):
#         assert lr > 0.0
#         assert all([0.0 <= beta <= 1.0 for beta in betas])
#         defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
#         super().__init__(params, defaults)
#         self.update_fn = update_fn_sign
#         if use_triton:
#             from lion_pytorch.triton import update_fn as triton_update_fn
#             self.update_fn = triton_update_fn
        
#     @torch.no_grad()
#     def step(self, closure: Optional[Callable] = None):
#         loss = None
#         if exists(closure):
#             with torch.enable_grad():
#                 loss = closure()
#         self.elapsed_time = 0
#         for group in self.param_groups:
#             for p in filter(lambda p: exists(p.grad), group["params"]):
#                 grad, lr, wd, beta1, beta2, state = p.grad, group["lr"], group["weight_decay"], *group["betas"], self.state[p]
#                 # init state - exponential moving average of gradient values
#                 if len(state) == 0:
#                     state["exp_avg"] = torch.zeros_like(p)
#                 exp_avg = state["exp_avg"]
#                 self.elapsed_time += update_fn_sign(p, grad, exp_avg, lr, wd, beta1, beta2)
#         return loss
