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


class PartialDiceLoss(DiceLoss):
    def __init__(self, **kwargs):
        super().__init__(idk=[1], **kwargs)


# ==== Focal Loss =======
# Focal loss: handles the cases where CE loss performs badly, namely
# 1. When class imbalance inherits bias in the process (majority class examples will dominate the loss function and gradient descent)
# 2. CE loss fails to distinguish between hard and easy examples. CE loss fails to pay more attention to hard examples

# --> focal loss focuses on the examples that the model gets wrong rather than the ones it can confidentially predict
# --> ensures that predictions on hard examples improve over time rather than becoming overly cofident with easy ones.
# Down Weighting: technique that reduces the influence of easy examples on the loss function > pays more attention on hard examples
# Focal Loss adds a modulating factor to the CE loss.
# FocalLoss = - alpha_i (i - p_i)^gamma * log(p_i)

class BinaryFocalLoss():
    def __init__(self, cross_entropy, gamma=2, alpha=0.25, **kwargs):
        """
        Focal Loss for binary classification using the provided CrossEntropy implementation from above.
        gamma: focusing parameter to control how much to focus on hard example
        alpha: balancing factor for class imbalance
        """
        self.gamma = gamma
        self.alpha = alpha
        self.cross_entropy = cross_entropy # CrossEntropy instance 
        print(f"Initialized {self.__class__.__name__} with gamma={self.gamma}, alpha={self.alpha}")

    def __call__(self, pred_softmax, weak_target):
        """
        pred_softmax: the predicted softmax probabilities
        weak_target: the target mask, containing binary labels (0 or 1)
        """

        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        # Compute the base CE loss using the existing CrossEntropy class
        # (This will compute: loss = - einsum("bkwh,bkwh->", mask, log_p))
        ce_loss = self.cross_entropy(pred_softmax, weak_target)

        # Get the probability of the true class
        prob_true = pred_softmax * weak_target + (1 - pred_softmax) * (1 - weak_target)

        # Calculate focal weight: (1 - prob_true)^gamma
        focal_weight = (1 - prob_true) ** self.gamma

        # Apply the alpha balancing factor
        alpha_factor = weak_target * self.alpha + (1 - weak_target) * (1 - self.alpha)

        # Apply alpha balancing factor to focal weight
        focal_loss = alpha_factor * focal_weight * ce_loss

        return focal_loss.mean() # not sure if I should use .mean()
