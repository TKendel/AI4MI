import torch 
import numpy as np
from utils import count_slices_per_patient, return_volumes, dice_batch, iou_batch
from scipy.spatial.distance import directed_hausdorff
import seg_metrics.seg_metrics as sg


def volume_dice(predictions, gts, path_to_slices):
    """
    Compute the Dice coefficient between predicted and ground truth volumes for each patient.
    Returns:dice_scores_per_patient: Dictionary where each key is a patient ID and the value is a tensor
    of Dice scores per organ for that patient.
    """
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
    """
    Compute the Intersection over Union (IoU) between predicted and ground truth volumes for each patient.
    Returns: iou_scores_per_patient: Dictionary where each key is a patient ID and the value is a tensor
    of IoU scores per organ for that patient.
    """
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



def volume_hausdorff(predictions, gts, path_to_slices, K, hd_95=False):
    """
    Compute the Hausdorff or 95th percentile Hausdorff (HD95) distance 
    between predicted and ground truth 3D volumes for each patient and each organ class.
    Returns: hausdorff_per_patient: Dictionary where each key is a patient ID and the value is a tensor
    of maximum Hausdorff distances per organ for that patient.
    """
    prediction_patient_volumes, gt_patient_volumes = return_volumes(predictions, gts, path_to_slices)
    hausdorff_per_patient = {}

    # Loop over each patient 
    for (patient_pred, volumepred), (patient_gt, volumegt) in zip(prediction_patient_volumes.items(), gt_patient_volumes.items()):
        assert patient_pred == patient_gt, "Mismatch in patient prediction and ground truth"
        
        # Get the diagonal distance as the maximum possible distance for the 3D volume (slices x H x W)
        slices, classes, H, W = volumepred.shape
        max_diagonal_distance = np.sqrt(slices**2 + H**2 + W**2)
        
        patient_hd = []

        # Loop over each organ class (skipping background class 0)
        for class_idx in range(1, K):
            pred_volume = volumepred[:, class_idx, :, :].numpy()
            gt_volume = volumegt[:, class_idx, :, :].numpy()

            # Extract boundary points (non-zero elements)
            pred_boundary = np.argwhere(pred_volume)
            gt_boundary = np.argwhere(gt_volume)

            assert torch.sum(pred_boundary) == pred_boundary.shape[0], "Mismatch in pred_slice: sum and shape don't match"
            assert  torch.sum(gt_boundary)== gt_boundary.shape[0], "Mismatch in gt_slice: sum and shape don't match"

             # Check if both volumes are empty (no segmentation)
            if pred_boundary.size == 0 and gt_boundary.size == 0:
                hausdorff_distance_class = 0.0
            elif pred_boundary.size > 0 and gt_boundary.size > 0:
                # Compute HD95 or HD using seg-metrics
                metrics = sg.write_metrics(
                    labels=[1],  # Because we ahve binary volumes
                    gdth_img=gt_volume,
                    pred_img=pred_volume,
                    metrics=['hd95'] if hd_95 else ['hd']  # Choose HD95 or HD
                )
                hausdorff_distance_class = metrics[0]['hd95'][0] if hd_95 else metrics[0]['hd'][0]
            else:
                # If one volume is empty, use the max diagonal distance as HD
                hausdorff_distance_class = max_diagonal_distance 
            
            patient_hd.append(hausdorff_distance_class)

        hausdorff_per_patient[patient_pred] = torch.tensor(patient_hd, dtype=torch.float32)
    return hausdorff_per_patient




# Because the hausdorff on volume works, we will no longer use the slice based one
def slice_hausdorff(predictions, gts, path_to_slices, K):
    """
    Compute the Hausdorff distance between predicted and ground truth slices
    for each patient, organ, and slice. Returns the maximum Hausdorff distance
    per organ for each patient. When one of the segmentations was empty, hc was set to the maximum distance.
    Because there were many such cases, we do not use this metric.
    """
    prediction_patient_volumes, gt_patient_volumes = return_volumes(predictions, gts, path_to_slices)
    hausdorff_per_patient = {}
    for (patient_pred, volumepred), (patient_gt, volumegt) in zip(prediction_patient_volumes.items(), gt_patient_volumes.items()):
        assert patient_pred == patient_gt, "Mismatch in patient data."
        
        # shape [nr slices, 5, 256, 256]
        H, W = volumepred.shape[2], volumepred.shape[3]
        max_distance = np.sqrt(H**2 + W**2)  # Max possible distance in a 2D slice
        
        patient_hd = []
        total_inf_count = 0  
        total_zero_count = 0 

        # Loop over each organ class (skipping background class 0)
        for class_idx in range(1,K):
            organ_hd = []
            inf_count = 0
            zero_count = 0
            
            # Extract organ segmentation volumes for the current class (organ)
            pred_volume = volumepred[:, class_idx, :, :]  
            gt_volume = volumegt[:, class_idx, :, :]     
            
            # Loop over each slice for this organ
            for slice_idx in range(pred_volume.shape[0]):
                pred_slice = pred_volume[slice_idx, :, :]
                gt_slice = gt_volume[slice_idx, :, :]
                sum_pred = torch.sum(pred_slice)
                sum_gt = torch.sum(gt_slice)

                # Extract boundary points (non-zero elements)
                pred_points = np.argwhere(pred_slice.numpy())
                gt_points = np.argwhere(gt_slice.numpy())

                # Verify the sum and shape match
                assert sum_pred.item() == pred_points.shape[0], "Mismatch in pred_slice: sum and shape don't match"
                assert sum_gt.item() == gt_points.shape[0], "Mismatch in gt_slice: sum and shape don't match"
            

                if pred_points.shape[0] == 0 and gt_points.shape[0] == 0:
                    hd = 0 # Both predictions and GT are empty
                    zero_count += 1
                elif pred_points.shape[0] == 0 or gt_points.shape[0] == 0:
                    hd = max_distance # One of the segmentations is empty
                    inf_count += 1
                else:
                    # Compute forward and reverse Hausdorff distances
                    assert pred_points.shape[1] == 2, f"Error: pred_points has incorrect shape {pred_points.shape}"
                    assert gt_points.shape[1] == 2, f"Error: gt_points has incorrect shape {gt_points.shape}"

                    hd_forward = directed_hausdorff(pred_points, gt_points)[0]
                    hd_reverse = directed_hausdorff(gt_points, pred_points)[0]
                    hd = max(hd_forward, hd_reverse)
                
                organ_hd.append(hd)
        
            total_inf_count += inf_count
            total_zero_count += zero_count
            # Append the maximum HD for this organ across all slices
            patient_hd.append(max(organ_hd))

        #print(f"Patient {patient_pred}: Total {total_zero_count} slices set to 0, {total_inf_count} slices set to inf.")
        hausdorff_per_patient[patient_pred] = torch.tensor(patient_hd)

    return hausdorff_per_patient