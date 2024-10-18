import torch
from torch import einsum
from utils import simplex, sset


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
        loss = - einsum("bkwh,bkwh->bkwh", mask, log_p)  # Keep the spatial dimensions for focal loss use
        if not focal_loss:
            loss /= mask.sum() + 1e-10

        return loss  # Return pixel-wise loss, not the reduced sum


class FocalLoss():
    def __init__(self, cross_entropy, gamma=2, alpha=[0.75, 0.25, 0.75, 0.25], **kwargs):
        """
        Arguments:
        - cross_entropy: An instance of CrossEntropy for computing CE loss
        - gamma: Focusing parameter for the focal loss
        - alpha: A list of balancing factors for class imbalance, one value per organ
        """
        self.cross_entropy = cross_entropy
        self.gamma = gamma
        self.alpha = alpha  # List of alpha values for each organ (e.g., [0.75, 0.25, 0.75, 0.25])
        self.idk = kwargs['idk']  # Indexes for the organs
        
        print(f"Initialized {self.__class__.__name__} with gamma={self.gamma}, alpha={self.alpha}")
        print(f"IDK (Organ Indexes): {self.idk}")

        # Ensure that alpha matches the number of organs in idk
        assert len(self.alpha) == len(self.idk), f"Length of alpha ({len(self.alpha)}) must match number of organs ({len(self.idk)})."

    def __call__(self, pred_softmax, weak_target):
        """
        Arguments:
        - pred_softmax: The predicted softmax probabilities from the model
        - weak_target: The ground truth binary target mask, containing binary labels (0 or 1)
        """
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        # Compute the pixel-wise CE loss (not reduced yet) using CrossEntropyForFocal
        pixelwise_ce_loss = self.cross_entropy(pred_softmax, weak_target)

        epsilon = 1e-6  # or some small value to avoid log(0) issues
        focal_loss = 0  # Initialize total focal loss

        # Loop through each organ index and apply the corresponding alpha and focal loss calculation
        for i, organ_idx in enumerate(self.idk):
            print(f"Processing organ index: {organ_idx} with alpha: {self.alpha[i]}")  # Debug print

            # Extract the relevant class's prediction and target
            pred_organ = pred_softmax[:, organ_idx, ...]
            target_organ = weak_target[:, organ_idx, ...]

            # Compute prob_true: pixel-wise probabilities of correct predictions for this organ
            prob_true = torch.clamp(pred_organ * target_organ + (1 - pred_organ) * (1 - target_organ),
                                    min=epsilon, max=1.0 - epsilon)

            # Focal weight: (1 - prob_true) ** gamma
            focal_weight = (1 - prob_true) ** self.gamma

            # Alpha factor: Use the organ-specific alpha from the list
            alpha_factor = target_organ * self.alpha[i] + (1 - target_organ) * (1 - self.alpha[i])

            # Pixel-wise focal loss = alpha_factor * focal_weight * pixelwise_ce_loss for this organ
            organ_focal_loss = alpha_factor * focal_weight * pixelwise_ce_loss[:, organ_idx, ...]

            # Accumulate the loss for each organ
            focal_loss += organ_focal_loss.mean()

        # Return the average focal loss across all organs
        return focal_loss / len(self.idk)



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
