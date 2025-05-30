import torch
import torch.nn as nn

class IoULoss3D(nn.Module):
    """
    Simple axis-aligned 3D IoU loss based on corner inputs.
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:

        pred_min = pred.min(dim=1).values  
        pred_max = pred.max(dim=1).values
        gt_min   = gt.min(dim=1).values
        gt_max   = gt.max(dim=1).values

        inter_min = torch.max(pred_min, gt_min)
        inter_max = torch.min(pred_max, gt_max)
        inter_dims = (inter_max - inter_min).clamp(min=0)  
        inter_vol  = inter_dims.prod(dim=1)               

        pred_vol = (pred_max - pred_min).clamp(min=0).prod(dim=1)
        gt_vol   = (gt_max   - gt_min).clamp(min=0).prod(dim=1)

        union = pred_vol + gt_vol - inter_vol + self.eps
        ious  = inter_vol / union

        return (1.0 - ious).mean()
