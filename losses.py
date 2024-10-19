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


from torch import einsum

from utils import simplex, sset


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

        loss = - einsum("bkwh,bkwh->", mask, log_p)
        loss /= mask.sum() + 1e-10

        return loss


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

class GeneralizedDice():
    def __init__(self, **kwargs):
        self.idk = kwargs['idk']
        self.smooth = 1e-10  # Smoothing factor to avoid division by zero
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax, target):
        assert pred_softmax.shape == target.shape
        assert simplex(pred_softmax)
        assert sset(target, [0, 1])

        pred = pred_softmax[:, self.idk, ...].float()
        target = target[:, self.idk, ...].float()

        weight = 1 / ((einsum("bkwh->bk", target) + 1e-10) ** 2)
        intersection = weight * einsum("bkwh,bkwh->bk", pred, target)
        union = weight * (einsum("bkwh->bk", pred) + einsum("bkwh->bk", target))

        divided = 1 - 2 * (einsum("bk->b", intersection) + 1e-10) / (einsum("bk->b", union) + 1e-10)

        loss = divided.mean()

        return loss
    
class CombinedLoss:
    def __init__(self, alpha=0.5, beta=0.5, **kwargs):
        print(f"Initialized {self.__class__.__name__} with {kwargs}")
        self.alpha = alpha
        self.beta = beta
        self.ce_loss = CrossEntropy(**kwargs)
        self.dice_loss = DiceLoss(**kwargs)

    def __call__(self, pred_softmax, target):
        ce = self.ce_loss(pred_softmax, target)
        dice = self.dice_loss(pred_softmax, target)
        return self.alpha * ce + self.beta * dice


class PartialDiceLoss(DiceLoss):
    def __init__(self, **kwargs):
        super().__init__(idk=[1], **kwargs)


class BinaryFocalLoss():
    def __init__(self, cross_entropy, gamma=2, alpha=0.25, **kwargs):
        """
        Focal Loss for binary classification using the CrossEntropy implementation from above.
        Arguments:
        - cross_entropy: The base cross-entropy loss instance to use
        - gamma: Focusing parameter to control how much to focus on hard example
        - alpha: Balancing factor to adjust for class imbalance (alpha for class 1, 1-alpha for class 0)
        """
        self.cross_entropy = cross_entropy 
        self.gamma = gamma
        self.alpha = alpha
        self.idk = kwargs['idk'] # Self.idk is used to filter out some classes of the target mask. Use fancy indexing
        
        print(f"Initialized {self.__class__.__name__} with gamma={self.gamma}, alpha={self.alpha}")

    def __call__(self, pred_softmax, weak_target):
        """
        Arguments:
        - pred_softmax: The predicted softmax probabilities from the model
        - weak_target: The ground truth binary target mask, containing binary labels (0 or 1)
        """

        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        # Compute the base CE loss by calling instance of the existing CrossEntropy class
        ce_loss = self.cross_entropy(pred_softmax, weak_target)

        # Get the probability of the true class for each pixel
        # since we're dealing with multiple classes and want to compute the loss only for certain organs, need to use [:, self.idk, ...]
        # self.idk contains the indices of the classes we're interested in, so we can compute the loss only for those classes
        prob_true = pred_softmax[:, self.idk, ...] * weak_target[:, self.idk, ...] + (1 - pred_softmax[:, self.idk, ...]) * (1 - weak_target[:, self.idk, ...])

        # Calculate focal weight: (1 - prob_true) ^ gamma
        focal_weight = (1 - prob_true) ** self.gamma

        # Apply the alpha balancing factor: α for class 1, (1 - α) for class 0
        alpha_factor = weak_target[:, self.idk, ...] * self.alpha + (1 - weak_target[:, self.idk, ...]) * (1 - self.alpha)

        # Now calculate the focal loss = alpha_factor * focal_weight * ce_loss
        focal_loss = alpha_factor * focal_weight * ce_loss

        # Optionally: apply normalization instead of taking the mean
        # Normalize focal_loss by the sum of the mask or relevant pixels
        normalized_loss = focal_loss / (weak_target[:, self.idk, ...].sum() + 1e-10)
        return normalized_loss # or normalized_loss.mean()

        # return focal_loss.mean() # reduce loss to a single scalar value
