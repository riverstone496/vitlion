import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftTargetCrossEntropy(nn.Module):
    def __init__(self, reduction='mean'):
        super(SoftTargetCrossEntropy, self).__init__()
        self.reduction = reduction

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            raise ValueError("Invalid reduction option. Only 'mean' and 'sum' are supported.")

