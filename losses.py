#!/usr/bin/env python3

# MIT License

# Copyright (c) 2024 Hoel Kervadec

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
from utils import simplex, sset
from torch import einsum
from torch import Tensor

class CrossEntropy():
    def __init__(self, **kwargs):
        # Self.idk is used to filter out some classes of the target mask. Use fancy indexing
        self.idk = kwargs['idk']
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax, weak_target):
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        log_p = (pred_softmax[:, self.idk, ...] + 1e-10).log()
        mask = weak_target[:, self.idk, ...].float()

        # Pixel-wise cross-entropy loss (not yet reduced)
        loss = -einsum("bkwh,bkwh->", mask, log_p)
        loss /= mask.sum() + 1e-10
        return loss  # Return pixel-wise loss, not the reduced sum

class PartialCrossEntropy(CrossEntropy):
    def __init__(self, **kwargs):
        super().__init__(idk=[1], **kwargs)

class DiceLoss():
    def __init__(self, **kwargs):
        # Self.idk is used to filter out some classes of the target mask. Use fancy indexing
        self.idk = kwargs['idk']
        self.smooth = 1e-10  # Smoothing factor to avoid division by zero
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax, target):
        """
            Computes the Dice loss for the given predictions and target mask.
        """
        assert pred_softmax.shape == target.shape
        assert simplex(pred_softmax)
        assert sset(target, [0, 1])

        pred = pred_softmax[:, self.idk, ...].float()
        target = target[:, self.idk, ...].float()

        intersection = einsum("bk...,bk...->bk", pred, target)
        pred_sum = einsum("bk...->bk", pred)
        target_sum = einsum("bk...->bk", target)

        dice_score = (2. * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)
        # Average Dice score across all batches and classes
        dice_score = dice_score.mean()

        # Check if the single Dice score is within the valid range [0, 1]
        if dice_score < 0 or dice_score > 1:
            raise ValueError(f"Dice score out of range! Dice score: {dice_score} - Terminating training.")

        # Dice Loss is 1 - Dice Coefficient
        loss = 1 - dice_score

        return loss
    
class PartialDiceLoss(DiceLoss):
    def __init__(self, **kwargs):
        super().__init__(idk=[1], **kwargs)


# This version is specifically for FocalLoss, without the reduction applied
class CrossEntropyPerClass():
    """
    Calculates cross-entropy loss per class to use for focal loss
    """
    def __init__(self, **kwargs):
        # Self.idk is used to filter out some classes of the target mask. Use fancy indexing
        self.idk = kwargs['idk']
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax, weak_target):
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        log_p = (pred_softmax[:, self.idk, ...] + 1e-10).log()
        mask = weak_target[:, self.idk, ...].float()  # mask shape torch.Size([8, 5, 256, 256])
        
        # Pixel-wise cross-entropy loss (not yet reduced)
        loss = -einsum("bkwh,bkwh->k", mask, log_p) 
        #mask_sum_per_class = mask.sum(dim=(0, 2, 3)) + 1e-10  # sum over batch (0), height (2), and width (3)  
        #loss /= mask_sum_per_class
        return loss  # Shape will be [num_classes], one value per class

class FocalLoss():
    """
    Calculates multiclass focal loss.
    Possibly with different alpha values if specified when calling to mitigate class imbalance.
    """
    def __init__(self, alpha=None, gamma=2, reduction='mean', **kwargs):
        self.alpha = alpha if alpha is not None else torch.ones(len(kwargs['idk'])) 
        self.gamma = gamma
        self.reduction = reduction
        self.ce_loss = CrossEntropyPerClass(**kwargs)
        print(f"Initialized {self.__class__.__name__} with alpha={self.alpha}, gamma={self.gamma}")

    def __call__(self, pred_softmax: Tensor, target: Tensor) -> Tensor:
        ce_loss = self.ce_loss(pred_softmax, target)
        p_t = torch.exp(-ce_loss)

        # Move self.alpha to the same device as p_t and ce_loss
        if isinstance(self.alpha, list):
            self.alpha = torch.tensor(self.alpha, dtype=p_t.dtype, device=p_t.device)
        else:
            self.alpha = self.alpha.to(p_t.device)  # Ensure alpha is on the same device

        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss


        # Reduce the loss based on the reduction method
        if self.reduction == 'mean':
            return focal_loss.mean()  # Mean reduction to scalar
        elif self.reduction == 'sum':
            return focal_loss.sum()   # Sum reduction to scalar
        else:
            return focal_loss         # No reduction (per-pixel loss returned)



class GeneralizedDice():
    def __init__(self, **kwargs):
        self.idk = kwargs['idk']
        self.smooth = 1e-10  # Smoothing factor to avoid division by zero
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax, target):
        """
            Computes the Generalized Dice loss.
            Uses class weights inversely proportional to the square of the target mask to calculate the loss.
        """
        assert pred_softmax.shape == target.shape
        assert simplex(pred_softmax)
        assert sset(target, [0, 1])

        pred = pred_softmax[:, self.idk, ...].float()
        target = target[:, self.idk, ...].float()

        # Calculate weight 
        weight = 1 / ((einsum("bkwh->bk", target) + 1e-10)**2) 
        # Calculate weighted dice score
        intersection = weight * einsum("bkwh,bkwh->bk", pred, target)
        union = weight * (einsum("bkwh->bk", pred) + einsum("bkwh->bk", target))

        dice_score = (2 * intersection + 1e-10) / (union + 1e-10)

        loss = 1- dice_score.mean()

        return loss
    

class CombinedLoss:
    def __init__(self, alpha=0.5, beta=0.5, **kwargs):
        print(f"Initialized {self.__class__.__name__} with {kwargs}")
        self.alpha = alpha
        self.beta = beta
        self.ce_loss = CrossEntropy(**kwargs)
        self.dice_loss = DiceLoss(**kwargs)

    def __call__(self, pred_softmax, target):
        """
            Computes a combined loss of CrossEntropy and Dice losses.
        """
        ce = self.ce_loss(pred_softmax, target)
        dice = self.dice_loss(pred_softmax, target)
        return self.alpha * ce + self.beta * dice


