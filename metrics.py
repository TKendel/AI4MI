from utils import count_slices_per_patient, dice_batch, iou_batch
from scipy.spatial.distance import cdist, directed_hausdorff
import numpy as np
import torch

def return_volumes(predictions, gts, path_to_slices):
    prediction_patient_volumes = {}
    gt_patient_volumes = {}
    start = 0
    count_slices = count_slices_per_patient(path_to_slices)
    # print("count slice dict", count_slices)
    for patient, num_slices in count_slices.items():
        # print("patient and slices order: ", patient, num_slices)
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
    # print("returned patients pred", prediction_patient_volumes.keys())
    # print("returned patients gt", gt_patient_volumes.keys())
    return prediction_patient_volumes, gt_patient_volumes



def volume_dice(predictions, gts, path_to_slices):
    prediction_patient_volumes, gt_patient_volumes = return_volumes(predictions, gts, path_to_slices)
    dice_scores_per_patient = {}
    for (patient_pred, volumepred), (patient_gt, volumegt) in zip(prediction_patient_volumes.items(), gt_patient_volumes.items()):
        assert patient_pred == patient_gt  # Ensure that we are processing the same patient
        dicescores = dice_batch(volumegt, volumepred)  #tensor([9.9693e-01, 1.2450e-12, 7.7337e-01, 1.2665e-12, 3.7515e-13]) --> tensor 5 classes 
        # Store the Dice scores per patient
        dice_scores_per_patient[patient_pred] = dicescores
    # Return the dictionary containing Dice scores for each patient
    return dice_scores_per_patient

def volume_iou(predictions, gts, path_to_slices):
    prediction_patient_volumes, gt_patient_volumes = return_volumes(predictions, gts, path_to_slices)

    iou_scores_per_patient = {}
    for (patient_pred, volumepred), (patient_gt, volumegt) in zip(prediction_patient_volumes.items(), gt_patient_volumes.items()):
            assert patient_pred == patient_gt  # Ensure that we are processing the same patient
            iou_scores = iou_batch(volumegt, volumepred)
            # Store the IOU scores per patient
            iou_scores_per_patient[patient_pred] = iou_scores
    # Return the dictionary containing IOU scores for each patient
    return iou_scores_per_patient



def volume_hausdorff(predictions, gts, path_to_slices, K, hd_95=False):
    prediction_patient_volumes, gt_patient_volumes = return_volumes(predictions, gts, path_to_slices)
    hausdorff_per_patient = {}
    for (patient_pred, volumepred), (patient_gt, volumegt) in zip(prediction_patient_volumes.items(), gt_patient_volumes.items()):
        assert patient_pred == patient_gt
        patient_hd = []

        # split sgementations by class to caluclate the haussdorf per organ
        for class_idx in range(5):
            pred_volume = volumepred[:, class_idx, :, :]
            gt_volume = volumegt[:, class_idx, :, :]

            # print(f"class: {class_idx}, shape pred {pred_volume.shape}")
            # print(f"class: {class_idx}, pred gt {gt_volume.shape}")

            # will return a list of coordinates, where each coordinate 
            #has the format [slice_index, row_index, column_index]
            pred_boundary = np.argwhere(pred_volume.cpu().numpy())
            gt_boundary = np.argwhere(gt_volume.cpu().numpy())

            if pred_boundary.size > 0 and gt_boundary.size > 0:
                if hd_95:
                    # compute pairwise distances 
                    distances_pred_to_gt = cdist(pred_boundary, gt_boundary, 'euclidean')
                    distances_gt_to_pred = cdist(gt_boundary, pred_boundary, 'euclidean')

                    # retrieve minimum distance 
                    min_distances_pred_to_gt = np.min(distances_pred_to_gt, axis=1)
                    min_distances_gt_to_pred = np.min(distances_gt_to_pred, axis=1)

                    # calculate the 95th percentile of the minimum distances
                    hd95_pred_to_gt = np.percentile(min_distances_pred_to_gt, 95)
                    hd95_gt_to_pred = np.percentile(min_distances_gt_to_pred, 95)

                    # The final HD95 is the maximum of the two 95th percentiles
                    hausdorff_distance_class = max(hd95_pred_to_gt, hd95_gt_to_pred)
                else:
                    forward_hausdorff = directed_hausdorff(pred_boundary, gt_boundary)[0]
                    backward_hausdorff = directed_hausdorff(gt_boundary, pred_boundary)[0]
                    hausdorff_distance_class = max(forward_hausdorff, backward_hausdorff)
            else:
                hausdorff_distance_class = np.nan  # Handle cases where one volume is empty (no segmentation)
            
            patient_hd.append(hausdorff_distance_class)
        hausdorff_per_patient[patient_pred] = torch.tensor(patient_hd, dtype=torch.float32)
    return hausdorff_per_patient
