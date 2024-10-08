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
        for class_idx in range(1, K):
            pred_volume = volumepred[:, class_idx, :, :]
            gt_volume = volumegt[:, class_idx, :, :]

            # print(f"class: {class_idx}, shape pred {pred_volume.shape}")
            # print(f"class: {class_idx}, shape gt {gt_volume.shape}")

            # print(f"Unique values in pred_volume: {torch.unique(pred_volume)}")
            # print(f"Unique values in gt_volume: {torch.unique(gt_volume)}")
            # unique_pred_values = torch.unique(pred_volume)
            # unique_gt_values = torch.unique(gt_volume)

            # # Check if either pred_volume or gt_volume contains only the value 0
            # if torch.equal(unique_pred_values, torch.tensor([0])) or torch.equal(unique_gt_values, torch.tensor([0])):
            #     print(f"Unique values in pred_volume: {unique_pred_values}")
            #     print(f"Unique values in gt_volume: {unique_gt_values}")


            # will return a list of coordinates, where each coordinate 
            #has the format [slice_index, row_index, column_index]
            pred_boundary = np.argwhere(pred_volume.numpy())
            gt_boundary = np.argwhere(gt_volume.numpy())

            assert torch.sum(pred_slice) == pred_boundary.shape[0], "Mismatch in pred_slice: sum and shape don't match"
            assert  torch.sum(gt_slice)== gt_boundary.shape[0], "Mismatch in gt_slice: sum and shape don't match"

            if pred_boundary.size == 0 and gt_boundary.size == 0:
                hausdorff_distance_class = 0.0
            elif pred_boundary.size > 0 and gt_boundary.size > 0:
                if hd_95:
                    # compute pairwise distances 
                    distances_pred_to_gt = cdist(pred_boundary, gt_boundary, 'euclidean')
                    distances_gt_to_pred = cdist(gt_boundary, pred_boundary, 'euclidean')

                    # retrieve minimum distance - for each point in the predicted boundary, it finds the nearest point in the ground truth boundary.
                    min_distances_pred_to_gt = np.min(distances_pred_to_gt, axis=1)  
                    min_distances_gt_to_pred = np.min(distances_gt_to_pred, axis=1)

                    # calculate the 95th percentile of the minimum distances
                    # --> finds a distance such that 95% of the minimum distances from predicted points to ground truth points are less than or equal to this value.
                    hd95_pred_to_gt = np.percentile(min_distances_pred_to_gt, 95)
                    hd95_gt_to_pred = np.percentile(min_distances_gt_to_pred, 95)

                    # The final HD95 is the maximum of the two 95th percentiles
                    hausdorff_distance_class = max(hd95_pred_to_gt, hd95_gt_to_pred)
                else:
                    forward_hausdorff = directed_hausdorff(pred_boundary, gt_boundary)[0]
                    backward_hausdorff = directed_hausdorff(gt_boundary, pred_boundary)[0]
                    hausdorff_distance_class = max(forward_hausdorff, backward_hausdorff)
            else:
                hausdorff_distance_class = np.nan  # Handle cases where one volume is empty (no segmentation) - not sure if this is the solution
            
            patient_hd.append(hausdorff_distance_class)
        #print("patient_hd", patient_hd)
        hausdorff_per_patient[patient_pred] = torch.tensor(patient_hd, dtype=torch.float32)
    return hausdorff_per_patient


def slicehausdorff(predictions, gts, path_to_slices, K):
    
    prediction_patient_volumes, gt_patient_volumes = return_volumes(predictions, gts, path_to_slices)
    hausdorff_per_patient = {}
    for (patient_pred, volumepred), (patient_gt, volumegt) in zip(prediction_patient_volumes.items(), gt_patient_volumes.items()):
        # both volumepred and volume gt have the shape [nr slices, 5, 256, 256]
        H, W = volumepred.shape[2], volumepred.shape[3]
        max_distance = np.sqrt(H**2 + W**2) 
        
        assert patient_pred == patient_gt
        patient_hd = []
        total_inf_count = 0  
        total_zero_count = 0 
        # Loop over each class (organ)
        for class_idx in range(1,K):
            organ_hd = []
            inf_count = 0
            zero_count = 0
            pred_volume = volumepred[:, class_idx, :, :]  # Extract the prediction for each organ across slices
            gt_volume = volumegt[:, class_idx, :, :]      # Extract the ground truth for each organ across slices
            # Loop over each slice
            for slice_idx in range(pred_volume.shape[0]):
                pred_slice = pred_volume[slice_idx, :, :]
                gt_slice = gt_volume[slice_idx, :, :]
                sum_pred = torch.sum(pred_slice)
                sum_gt = torch.sum(gt_slice)

                # Extract boundary points (non-zero elements)
                pred_points = np.argwhere(pred_slice.numpy())
                gt_points = np.argwhere(gt_slice.numpy())
                #print(pred_points.shape)

                # Verify the sum and shape match
                assert sum_pred.item() == pred_points.shape[0], "Mismatch in pred_slice: sum and shape don't match"
                assert sum_gt.item() == gt_points.shape[0], "Mismatch in gt_slice: sum and shape don't match"
            

                if pred_points.shape[0] == 0 and gt_points.shape[0] == 0:
                    #print(f"Both pred_slice and gt_slice are empty for slice {slice_idx}")
                    hd = 0
                    zero_count += 1
                elif pred_points.shape[0] == 0 or gt_points.shape[0] == 0:
                    #print(f"One of pred_slice or gt_slice is empty for slice {slice_idx}")
                    hd = np.inf
                    inf_count += 1

                else:
                    assert pred_points.shape[1] == 2, f"Error: pred_points has incorrect shape {pred_points.shape}"
                    assert gt_points.shape[1] == 2, f"Error: gt_points has incorrect shape {gt_points.shape}"

                    #print(f"Calculating Hausdorff distance for Class {class_idx}, Slice {slice_idx}")
                    hd_forward = directed_hausdorff(pred_points, gt_points)[0]
                    hd_reverse = directed_hausdorff(gt_points, pred_points)[0]
                    hd = max(hd_forward, hd_reverse)
                
                organ_hd.append(hd)
        
            #print(f"Patient {patient_pred}, Organ {class_idx}: {zero_count} slices set to 0, {inf_count} slices set to inf.")
            total_inf_count += inf_count
            total_zero_count += zero_count
            patient_hd.append(max(organ_hd))

        # Take the maximum Hausdorff distance for this patient
        #print(f"Patient {patient_pred}: Total {total_zero_count} slices set to 0, {total_inf_count} slices set to inf.")
        hausdorff_per_patient[patient_pred] = torch.tensor(patient_hd)

    return hausdorff_per_patient