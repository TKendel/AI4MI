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


def compute_chunked_cdist(tensor1, tensor2, chunk_size=1024):
    distances = []
    for i in range(0, tensor1.shape[0], chunk_size):
        chunk_distances = []
        tensor1_chunk = tensor1[i:i+chunk_size]
        for j in range(0, tensor2.shape[0], chunk_size):
            tensor2_chunk = tensor2[j:j+chunk_size]
            # Compute cdist for the chunk and append the results
            chunk_distances.append(torch.cdist(tensor1_chunk, tensor2_chunk))
        # Concatenate the results for the current chunk of tensor1
        distances.append(torch.cat(chunk_distances, dim=1))
    # Concatenate all chunk results along the first dimension
    return torch.cat(distances, dim=0)


def volume_hausdorff(predictions, gts, path_to_slices, K, hd_95=False):
    prediction_patient_volumes, gt_patient_volumes = return_volumes(predictions, gts, path_to_slices)
    hausdorff_per_patient = {}
    for (patient_pred, volumepred), (patient_gt, volumegt) in zip(prediction_patient_volumes.items(), gt_patient_volumes.items()):
        # Get the diagonal distance as the maximum possible distance for the 3D volume (slices x H x W)
        slices, classes, H, W = volumepred.shape
        max_diagonal_distance = np.sqrt(slices**2 + H**2 + W**2)
        assert patient_pred == patient_gt
        patient_hd = []

        # split sgementations by class to caluclate the haussdorf per organ
        for class_idx in range(1, K):
            pred_volume = volumepred[:, class_idx, :, :]
            gt_volume = volumegt[:, class_idx, :, :]

            # will return a list of coordinates, where each coordinate 
            # has the format [slice_index, row_index, column_index]
            pred_boundary = np.argwhere(pred_volume.numpy())
            gt_boundary = np.argwhere(gt_volume.numpy())

            assert torch.sum(pred_boundary) == pred_boundary.shape[0], "Mismatch in pred_slice: sum and shape don't match"
            assert  torch.sum(gt_boundary)== gt_boundary.shape[0], "Mismatch in gt_slice: sum and shape don't match"

            if pred_boundary.size == 0 and gt_boundary.size == 0:
                hausdorff_distance_class = 0.0
            elif pred_boundary.size > 0 and gt_boundary.size > 0:
                if hd_95:

                    #####################################################################################
                    # i tried this; using gpu, deleting it, chuncking and all did not work, it keeps giving me the memory error
                    # Convert boundaries to torch tensors and move to GPU
                    # pred_boundarygpu = torch.tensor(pred_boundary, dtype=torch.float32).cuda()
                    # print(f"shape pred_boundarygpu {pred_boundarygpu.shape}")
                    # gt_boundarygpu = torch.tensor(gt_boundary, dtype=torch.float32).cuda()
                    # print(f"shape gt_boundarygpu {gt_boundarygpu.shape}")

                    ## Compute pairwise distances 
                    # distances_pred_to_gt = torch.cdist(pred_boundarygpu, gt_boundarygpu)
                    # distances_gt_to_pred = torch.cdist(gt_boundarygpu, pred_boundarygpu)
                    # # could try chunking 
                    # # distances_pred_to_gt = compute_chunked_cdist(pred_boundarygpu, gt_boundarygpu, chunk_size=1024)
                    # # distances_gt_to_pred = compute_chunked_cdist(gt_boundarygpu, pred_boundarygpu, chunk_size=1024)

                    # # Find the minimum distance for each point in both directions
                    # min_distances_pred_to_gt = torch.min(distances_pred_to_gt, dim=1)[0]
                    # min_distances_gt_to_pred = torch.min(distances_gt_to_pred, dim=1)[0]

                    # # Combine distances and compute 95th percentile
                    # all_min_distances = torch.cat([min_distances_pred_to_gt, min_distances_gt_to_pred])
                    # hausdorff_distance_class = torch.quantile(all_min_distances, 0.95).item()

                    # del distances_pred_to_gt, distances_gt_to_pred, pred_boundarygpu, gt_boundarygpu
                    # torch.cuda.empty_cache()

                    #####################################################################################
                    # this is what i orignally had, but that does nto work either...
                    # error: /var/spool/slurm/slurmd/job8133928/slurm_script: line 23: 427636 Killed             
                    # >slurmstepd: error: Detected 1 oom_kill event in StepId=8133928.batch. Some of the step tasks have been OOM Killed.

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
                hausdorff_distance_class = max_diagonal_distance # Handle cases where one volume is empty (no segmentation) - not sure if this is the solution
            
            patient_hd.append(hausdorff_distance_class)
        #print("patient_hd", patient_hd)
        hausdorff_per_patient[patient_pred] = torch.tensor(patient_hd, dtype=torch.float32)
    return hausdorff_per_patient


def slice_hausdorff(predictions, gts, path_to_slices, K):
    
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

                # Verify the sum and shape match
                assert sum_pred.item() == pred_points.shape[0], "Mismatch in pred_slice: sum and shape don't match"
                assert sum_gt.item() == gt_points.shape[0], "Mismatch in gt_slice: sum and shape don't match"
            

                if pred_points.shape[0] == 0 and gt_points.shape[0] == 0:
                    #print(f"Both pred_slice and gt_slice are empty for slice {slice_idx}")
                    hd = 0
                    zero_count += 1
                elif pred_points.shape[0] == 0 or gt_points.shape[0] == 0:
                    #print(f"One of pred_slice or gt_slice is empty for slice {slice_idx}")
                    hd = max_distance
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