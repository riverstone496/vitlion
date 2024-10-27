
from typing import Callable, Optional, Tuple

import torch
from torch.optim.optimizer import Optimizer
from torch.distributed import all_reduce, ReduceOp
import time, os

# functions
def exists(val):
    return val is not None

def stochastic_quantization(v, s):
    # ベクトル v の L2ノルムを計算
    norm_v = torch.norm(v, p=2)
    
    # ゼロベクトルの場合はゼロを返す
    if norm_v == 0:
        return torch.zeros_like(v, dtype=torch.int32)  # 整数値のゼロを返す
    
    # 各要素の絶対値の比を計算
    abs_v_norm = torch.abs(v) / norm_v
    
    # 量子化の範囲を定義
    quantized_values = torch.floor(abs_v_norm * s).clamp(0, s - 1)
    # 確率的に次のレベルに量子化
    random_vals = torch.rand_like(v)
    p = abs_v_norm * s - quantized_values
    xi = torch.where(random_vals > p, quantized_values, quantized_values + 1)

    # 最終的な量子化結果を計算
    quantized_v = (torch.sign(v) * xi).to(torch.int32)  # 整数値の量子化を行う
    
    return quantized_v

# update functions
def compute_updates_sign_int8(params, grads, exp_avgs, lr, wd, beta1, s):
    """
    Compute the sign-based updates for all parameters and aggregate them into a single vector.
    """
    updates_sign = []
    updates = []
    for p, grad, exp_avg in zip(params, grads, exp_avgs):
        # Step weight decay
        p.data.mul_(1 - lr * wd)
        # Compute update: update = sign(beta1 * exp_avg + (1 - beta1) * grad)
        update_sign = stochastic_quantization(exp_avg.mul(beta1).add(grad, alpha=1 - beta1), s = s).to(torch.int8)
        updates_sign.append(update_sign.flatten())

        update = exp_avg.mul(beta1).add(grad, alpha=1 - beta1)
        updates.append(update.flatten())
    # Concatenate all updates into a single vector
    update_vector_sign = torch.cat(updates_sign)
    update_vector = torch.cat(updates)
    return update_vector_sign, update_vector

# class
class CMPSSignSignLion(Optimizer):
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
        self.s = int(127 / self.ngpus)

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
        update_vector, update_vector_full = compute_updates_sign_int8(params, grads, exp_avgs, lr, wd, beta1, s = self.s)

        # Perform all_reduce on the aggregated update vector
        if torch.distributed.is_initialized():
            torch.cuda.synchronize()  # Ensure all previous operations are complete
            start_time = time.time()
            all_reduce(update_vector, op=ReduceOp.SUM)
            torch.cuda.synchronize()  # Ensure all_reduce is complete
            elapsed_time = time.time() - start_time
            # Normalize the update if necessary (e.g., divide by world size)
            all_reduce(update_vector_full, op=ReduceOp.SUM)
            update_vector = update_vector.sign_()
            update_vector_full = update_vector_full.sign_()
            self.matches = (update_vector == update_vector_full).sum().item()
            self.un_matches = (update_vector == -update_vector_full).sum().item()
        else:
            elapsed_time = 0.0

        # Apply the updates back to each parameter
        idx = 0  # Pointer to traverse the update_vector
        update_matches = []
        update_unmatches = []
        for p, exp_avg in zip(params, exp_avgs):
            numel = p.numel()
            # Extract the corresponding update for this parameter
            update = update_vector[idx:idx + numel].view_as(p).to(original_dtype)
            update_full = update_vector_full[idx:idx + numel].view_as(p).to(original_dtype)
            update_matches.append(100 * (update == update_full).sum().item() / numel)
            update_unmatches.append(100 * (update == -update_full).sum().item() / numel)
            idx += numel
            # Apply the update
            p.add_(update, alpha=-lr)
            # Update the exponential moving average
            exp_avg.mul_(beta2).add_(p.grad, alpha=1 - beta2)

        self.elapsed_time = elapsed_time
        self.numel = idx
        self.matches_ratio = self.matches / self.numel
        self.update_matches = update_matches
        self.update_unmatches = update_unmatches
        self.update_matches_cor = calculate_ratios(update_vector, update_vector_full)
        return loss

def calculate_ratios(update_vector, update_vector_full):
    # 各条件に対応するカウントを保持する辞書
    counts = {
        "sign_matches_cor/(0, 0)": ((update_vector == 0) & (update_vector_full == 0)).sum().item(),
        "sign_matches_cor/(1, 1)": ((update_vector == 1) & (update_vector_full == 1)).sum().item(),
        "sign_matches_cor/(-1, -1)": ((update_vector == -1) & (update_vector_full == -1)).sum().item(),
        "sign_matches_cor/(1, 0)": ((update_vector == 1) & (update_vector_full == 0)).sum().item(),
        "sign_matches_cor/(1, -1)": ((update_vector == 1) & (update_vector_full == -1)).sum().item(),
        "sign_matches_cor/(-1, 0)": ((update_vector == -1) & (update_vector_full == 0)).sum().item(),
        "sign_matches_cor/(0, 1)": ((update_vector == 0) & (update_vector_full == 1)).sum().item(),
        "sign_matches_cor/(0, -1)": ((update_vector == 0) & (update_vector_full == -1)).sum().item(),
        "sign_matches_cor/(-1, 1)": ((update_vector == -1) & (update_vector_full == 1)).sum().item(),
        "sign_matches_cor/lion_0": ((update_vector_full == 0)).sum().item(),
        "sign_matches_cor/lion_1": ((update_vector_full == 1)).sum().item(),
        "sign_matches_cor/lion_-1":( (update_vector_full == -1)).sum().item(),
        "sign_matches_cor/sign_lion_0": ((update_vector == 0)).sum().item(),
        "sign_matches_cor/sign_lion_1": ((update_vector == 1)).sum().item(),
        "sign_matches_cor/sign_lion_-1":( (update_vector == -1)).sum().item()
    }

    # 総数を計算して割合を求める
    ratios = {key: 100 * count / update_vector.numel() for key, count in counts.items()}

    return ratios