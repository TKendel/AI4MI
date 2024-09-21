from utils import count_slices_per_patient, dice_batch
def volume_dice(predictions, gts, path_to_slices):
    """
    Computes Dice similarity coefficient for predicted and ground truth segmentations per patient.

    Parameters:
    - predictions (Tensor): Predicted segmentation volumes, shape (num_slices, num_classes, height, width).
    - gts (Tensor): Ground truth segmentation volumes, shape same as `predictions`.
    - path_to_slices (str): Path to directory containing patient slices, used to count slices per patient.

    Returns:
    - dice_scores_per_patient (dict): A dictionary where keys are patient IDs and values are tensors of Dice scores 
      for each class, shape (num_classes,).
    
    Workflow:
    1. Organizes slices into patient volumes.
    2. For each patient, computes the Dice score across all classes using `dice_batch`.
    3. Returns a dictionary mapping patient IDs to their corresponding Dice scores.
    """
    prediction_patient_volumes = {}
    gt_patient_volumes = {}
    dice_scores_per_patient = {}

    start = 0
    count_slices = count_slices_per_patient(path_to_slices)
    
    for patient, num_slices in count_slices.items():
        predictions_patient_slices = predictions[start:start + num_slices]
        gt_patient_slices = gts[start:start + num_slices]

        prediction_patient_volumes[patient] = predictions_patient_slices
        gt_patient_volumes[patient] = gt_patient_slices

        # {
        # "Patient_01": tensor of shape (nr slices, nr classes, 256, 256),
        # "Patient_02": tensor of shape (nr slices, nr classes, 256, 256),
        # "Patient_03": tensor of shape (nr slices, nr classes, 256, 256),
        # }
        
        # Update the start index for the next patient
        start += num_slices

    
    for (patient_pred, volumepred), (patient_gt, volumegt) in zip(prediction_patient_volumes.items(), gt_patient_volumes.items()):
        assert patient_pred == patient_gt  # Ensure that we are processing the same patient
        dicescores = dice_batch(volumegt, volumepred)
    
        # Store the Dice scores per patient
        dice_scores_per_patient[patient_pred] = dicescores
        print(dice_scores_per_patient)

    # Return the dictionary containing Dice scores for each patient
    return dice_scores_per_patient