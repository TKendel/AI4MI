import torch 
import numpy as np
from utils import return_volumes, dice_batch, iou_batch
import seg_metrics.seg_metrics as sg
from skimage.morphology import skeletonize


def volume_dice(predictions, gts, path_to_slices):
    '''
    Compute the Dice coefficient between predicted and ground truth volumes for each patient.
    Returns:dice_scores_per_patient: Dictionary where each key is a patient ID and the value is a tensor
    of Dice scores per organ for that patient.
    '''
    prediction_patient_volumes, gt_patient_volumes = return_volumes(predictions, gts, path_to_slices)
    dice_scores_per_patient = {}

    # Loop over each patient
    for (patient_pred, volumepred), (patient_gt, volumegt) in zip(prediction_patient_volumes.items(), gt_patient_volumes.items()):
        # Ensure that we are processing the same patient
        assert patient_pred == patient_gt, "Mismatch in patient prediction and ground truth"
        
        # Calculate Dice score for each organ class (including background)
        dicescores = dice_batch(volumegt, volumepred)  
        # Store the Dice scores per patient
        dice_scores_per_patient[patient_pred] = dicescores
    return dice_scores_per_patient


def volume_iou(predictions, gts, path_to_slices):
    '''
    Compute the Intersection over Union (IoU) between predicted and ground truth volumes for each patient.
    Returns: iou_scores_per_patient: Dictionary where each key is a patient ID and the value is a tensor
    of IoU scores per organ for that patient.
    '''
    prediction_patient_volumes, gt_patient_volumes = return_volumes(predictions, gts, path_to_slices)
    iou_scores_per_patient = {}

    # Loop over each patient 
    for (patient_pred, volumepred), (patient_gt, volumegt) in zip(prediction_patient_volumes.items(), gt_patient_volumes.items()):
        # Ensure that we are processing the same patient
        assert patient_pred == patient_gt, "Mismatch in patient prediction and ground truth"
        # Calculate IoU score for each organ class (including background)
        iou_scores = iou_batch(volumegt, volumepred)
        # Store the IOU scores per patient
        iou_scores_per_patient[patient_pred] = iou_scores
    return iou_scores_per_patient


def distance_based_metrics(predictions, gts, path_to_slices, K):
    '''
    Compute distance-based metrics (Hausdorff Distance, 95th percentile Hausdorff Distance, 
    and Average Surface Distance) for each class (organ) across patient volumes.

    Returns:
        hausdorff_per_patient (dict): Hausdorff distance for each organ per patient.
        _95hausdorff_per_patient (dict): 95th percentile Hausdorff distance for each organ per patient.
        asd_per_patient (dict): Average Surface Distance (ASD) for each organ per patient.
    '''
    prediction_patient_volumes, gt_patient_volumes = return_volumes(predictions, gts, path_to_slices)
    hausdorff_per_patient = {}
    _95hausdorff_per_patient = {}
    asd_per_patient = {}

    # Loop over each patient 
    for (patient_pred, volumepred), (patient_gt, volumegt) in zip(prediction_patient_volumes.items(), gt_patient_volumes.items()):
        assert patient_pred == patient_gt, "Mismatch in patient prediction and ground truth"
        
        # Get the diagonal distance as the maximum possible distance for the 3D volume (slices x H x W)
        slices, classes, H, W = volumepred.shape
        max_diagonal_distance = np.sqrt(slices**2 + H**2 + W**2)
        
        patient_hd = []
        patient_95hd =[]
        patient_asd = []

        # Loop over each organ class (skipping background class 0)
        for class_idx in range(1, K):
            pred_volume = volumepred[:, class_idx, :, :].numpy()
            gt_volume = volumegt[:, class_idx, :, :].numpy()
  
            # Check if both volumes are empty (no segmentation) 
            if np.sum(pred_volume) == 0 and np.sum(gt_volume) == 0:
                hausdorff_distance_class = 0.0
                _95hausdorff_distance_class = 0.0
                asd_class = 0.0
            # If one volume is empty, use the max diagonal distance
            elif np.sum(pred_volume) == 0 or np.sum(gt_volume) == 0:
                hausdorff_distance_class = max_diagonal_distance
                _95hausdorff_distance_class = max_diagonal_distance
                asd_class = max_diagonal_distance
            else:
                metrics = sg.write_metrics(
                    labels=[1],  # Because we have binary volumes
                    gdth_img=gt_volume,
                    pred_img=pred_volume,
                    metrics=['hd', 'hd95', 'msd'] 
                )
                hausdorff_distance_class = metrics[0]['hd'][0] 
                _95hausdorff_distance_class = metrics[0]['hd95'][0]
                asd_class = metrics[0]['msd'][0]

            patient_asd.append(asd_class)
            patient_hd.append(hausdorff_distance_class)
            patient_95hd.append(_95hausdorff_distance_class)

        hausdorff_per_patient[patient_pred] =  torch.tensor(patient_hd, dtype=torch.float32)
        _95hausdorff_per_patient[patient_pred] = torch.tensor(patient_95hd, dtype=torch.float32)
        asd_per_patient[patient_pred] = torch.tensor(patient_asd, dtype=torch.float32)
    return hausdorff_per_patient, _95hausdorff_per_patient, asd_per_patient


def cl_score(v, s):
    return np.sum(v * s) / np.sum(s)


def cldice(predictions, gts, path_to_slices):
    prediction_patient_volumes, gt_patient_volumes = return_volumes(predictions, gts, path_to_slices)
    cldice_per_patient = {}
    # Loop over each patient 
    for (patient_pred, volumepred), (patient_gt, volumegt) in zip(prediction_patient_volumes.items(), gt_patient_volumes.items()):
        assert patient_pred == patient_gt, "Mismatch in patient prediction and ground truth"

        cldice_patient = []
        # Only calculate this score for esophagus and aorta
        for class_idx in [1,4]:
            pred_volume = volumepred[:, class_idx, :, :].numpy()  
            gt_volume = volumegt[:, class_idx, :, :].numpy()
            
            # Check if both volumes are empty (no segmentation)
            if np.sum(pred_volume) == 0 and np.sum(gt_volume) == 0:
                cldice_score = 1.0  # Both empty, perfect match 
            # If one volume is empty, use worst score
            elif np.sum(pred_volume) == 0 or np.sum(gt_volume) == 0:
                cldice_score = 0.0
            else:
                tprec = cl_score(gt_volume,skeletonize(pred_volume)) # susceptible to false positives 
                tsens = cl_score(pred_volume,skeletonize(gt_volume)) # susceptible to false negatives.
                if tprec + tsens == 0:
                    cldice_score = 0.0
                else:
                    cldice_score = 2 * ((tprec * tsens) / (tprec + tsens))
            cldice_patient.append(cldice_score)
        cldice_per_patient[patient_pred] = torch.tensor(cldice_patient, dtype=torch.float32)
    return cldice_per_patient
