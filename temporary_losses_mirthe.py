import torch
from torch import einsum
from utils import simplex, sset
from torch import Tensor

# This version is specifically for FocalLoss, without the reduction applied.
class CrossEntropy():
    def __init__(self, **kwargs):
        # Self.idk is used to filter out some classes of the target mask. Use fancy indexing
        self.idk = kwargs['idk']
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax, weak_target, focal_loss=False):
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        log_p = (pred_softmax[:, self.idk, ...] + 1e-10).log()
        mask = weak_target[:, self.idk, ...].float()

        # Pixel-wise cross-entropy loss (not yet reduced)
        loss = - einsum("bkwh,bkwh->", mask, log_p)  # Keep the spatial dimensions for focal loss use
        if not focal_loss:
            loss /= mask.sum() + 1e-10

        return loss  # Return pixel-wise loss, not the reduced sum

class FocalLoss():
    def __init__(self, alpha=0.25, gamma=2, reduction='mean', **kwargs):
        self.alpha = alpha 
        self.gamma = gamma
        self.reduction = reduction
        # self.cross_entropy = cross_entropy # Uses the already existing instance of CrossEntropy
        self.ce_loss = CrossEntropy(**kwargs)

        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax: Tensor, target: Tensor) -> Tensor:
        ce_loss = self.ce_loss(pred_softmax, target, focal_loss=True)
        p_t = torch.exp(-ce_loss) # probability of the true clas (exp(-cross entropy))

        # focal loss: alpha * (1 - pt)^gamma * CE
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss

        # Apply reduction: mean or sum
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss # no reduction (element-wise loss)


class DiceLoss():
    def __init__(self, **kwargs):
        # Self.idk is used to filter out some classes of the target mask. Use fancy indexing
        self.idk = kwargs['idk']
        self.smooth = 1e-10  # Smoothing factor to avoid division by zero
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax, target):
        assert pred_softmax.shape == target.shape
        assert simplex(pred_softmax)
        assert sset(target, [0, 1])

        pred = pred_softmax[:, self.idk, ...].float()
        target = target[:, self.idk, ...].float()

        intersection = einsum("bk...,bk...->bk", pred, target)
        pred_sum = einsum("bk...->bk", pred)
        target_sum = einsum("bk...->bk", target)

        dice_score = (2. * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)

        # Dice Loss is 1 - Dice Coefficient
        loss = 1 - dice_score.mean()

        return loss
