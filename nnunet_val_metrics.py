import torch 
import numpy as np
from utils import count_slices_per_patient, return_volumes, dice_batch, iou_batch
from scipy.spatial.distance import directed_hausdorff
import seg_metrics.seg_metrics as sg
from skimage.morphology import skeletonize


import nibabel as nib
import os
from sklearn.metrics import confusion_matrix

import tabulate




















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


def distance_based_metrics(predictions, gts, path_to_slices, K):
    """
    Compute distance-based metrics (Hausdorff Distance, 95th percentile Hausdorff Distance, 
    and Average Surface Distance) for each class (organ) across patient volumes.

    Returns:
        hausdorff_per_patient (dict): Hausdorff distance for each organ per patient.
        _95hausdorff_per_patient (dict): 95th percentile Hausdorff distance for each organ per patient.
        asd_per_patient (dict): Average Surface Distance (ASD) for each organ per patient.
    """
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



















#get overlap
"""
overlap_tp = np.where((gt_np == predicted_np and gt_np!=0).all, predicted, 0)  #where they match, and gt is NOT zero
overlap_tn = np.where(np.all((gt_np == predicted_np and gt_np==0)), predicted, 0)  #where they match, and gt is zero
overlap_fp = np.where(np.all((gt_np != predicted_np and gt_np==0)), predicted, 0)  #where they dont match, and the gt is zero; a fp was asserted
overlap_fn = np.where(np.all((gt_np != predicted_np and gt_np!=0)), predicted, 0)  #where they dont match, and the gt is NOT zero; a fn was asserted
"""

"""
overlap_tp = np.where((gt_np == predicted_np), predicted, 0)  #where they match, and gt is NOT zero
overlap_tn = np.where((gt_np == predicted_np ), predicted, 0)  #where they match, and gt is zero
overlap_fp = np.where((gt_np != predicted_np ), predicted, 0)  #where they dont match, and the gt is zero; a fp was asserted
overlap_fn = np.where((gt_np != predicted_np ), predicted, 0)  #where they dont match, and the gt is NOT zero; a fn was asserted
"""
















def cldice2(predictions, gts):
    prediction_patient_volumes = predictions
    gt_patient_volumes = gts
    cldice_per_patient = {}
    # Loop over each patient 
    #for (patient_pred, volumepred), (patient_gt, volumegt) in zip(prediction_patient_volumes.items(), gt_patient_volumes.items()):
    #assert patient_pred == patient_gt, "Mismatch in patient prediction and ground truth"

    volumepred = predictions
    volumegt = gts

    cldice_patient = []
    # Only calculate this score for esophagus and aorta
    for class_idx in range(4):
        # one-hot encoded volumes of each class
        pred_volume = volumepred[class_idx]
        gt_volume = volumegt[class_idx]
        
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
    #cldice_per_patient[patient_pred] = torch.tensor(cldice_patient, dtype=torch.float32)

    return cldice_patient #cldice_per_patient



def DICE(predicted, gt):
    """
    comb = predicted + gt
    dif = np.where(gt == 2, 1, 0)

    tp = np.where(comb == 2, 1, 0)
    fp = 
    """
    #cm = confusion_matrix(gt, predicted) # sklearn.metrics.confusion_matrix(y_true, y_pred, *, labels=None, sample_weight=None, normalize=None)
    tp_ = np.where(predicted == gt, 1, 0)
    comb = gt - (predicted*5)   # 0 is no, 1 or 5 is yes
    # if both are 0 -- 0-0 = 0 TN
    # if gt is 1, pred 5 -- 1-5 = -4 TP
    # if gt is 0, pred 5 -- 0-5 = -5 FP
    # if gt is 1, pred 0 -- 1-0 = 1 FN
    tp = np.where(comb == -4, 1, 0)
    tn = np.where(comb == 0, 1, 0)
    fp = np.where(comb == -5, 1, 0)
    fn = np.where(comb == 1, 1, 0)

    dice = (2*np.sum(tp))/((2*np.sum(tp))+np.sum(fp)+np.sum(fn))

    return dice




def volume_dice(predictions, gts):
    """
    Compute the Dice coefficient between predicted and ground truth volumes for each patient.
    Returns:dice_scores_per_patient: Dictionary where each key is a patient ID and the value is a tensor
    of Dice scores per organ for that patient.
    """
    prediction_patient_volumes = predictions
    gt_patient_volumes = gts
    dice_scores_per_patient = {}

    dice_per_class = []

    # Loop over each patient
    #for (patient_pred, volumepred), (patient_gt, volumegt) in zip(prediction_patient_volumes.items(), gt_patient_volumes.items()):
    # Ensure that we are processing the same patient
    #assert patient_pred == patient_gt, "Mismatch in patient prediction and ground truth"
    
    for i in range(4):
        # Calculate Dice score for each organ class (including background)
        pred_torch = torch.from_numpy(prediction_patient_volumes[i])
        gt_torch = torch.from_numpy(gt_patient_volumes[i])
        #dicescores = dice_batch(pred_torch, gt_torch)  #one hot error,
        dicescores = DICE(prediction_patient_volumes[i], gt_patient_volumes[i])
        # Store the Dice scores per patient
        dice_per_class.append(dicescores)


    return dice_per_class #dice_scores_per_patient




def distance_metric(gts, preds):

    gt_list = gts
    pred_list = preds

    H, W, slices= predicted_np.shape
    max_diagonal_distance = np.sqrt(slices**2 + H**2 + W**2)

    classes = K = 4

    patient_hd = []
    patient_95hd =[]
    patient_asd = []

    for class_idx in range(0, K):
        #pred_volume = predicted_np[:, class_idx, :, :].numpy()
        #gt_volume = gt_np[:, class_idx, :, :].numpy()

        # Check if both volumes are empty (no segmentation) 
        if np.sum(pred_list[class_idx]) == 0 and np.sum(gt_list[class_idx]) == 0:
            hausdorff_distance_class = 0.0
            _95hausdorff_distance_class = 0.0
            asd_class = 0.0
        # If one volume is empty, use the max diagonal distance
        elif np.sum(pred_list[class_idx]) == 0 or np.sum(gt_list[class_idx]) == 0:
            hausdorff_distance_class = max_diagonal_distance
            _95hausdorff_distance_class = max_diagonal_distance
            asd_class = max_diagonal_distance
        else:
            metrics = sg.write_metrics(
                labels=[1],  # Because we have binary volumes
                gdth_img=gt_list[class_idx],
                pred_img=pred_list[class_idx],
                metrics=['hd', 'hd95', 'msd'] 
            )
            hausdorff_distance_class = metrics[0]['hd'][0] 
            _95hausdorff_distance_class = metrics[0]['hd95'][0]
            asd_class = metrics[0]['msd'][0]

        patient_asd.append(asd_class)
        patient_hd.append(hausdorff_distance_class)
        patient_95hd.append(_95hausdorff_distance_class)

    return(patient_hd, patient_95hd, patient_asd)


def idv_iou(gt, predicted):
    comb = gt - (predicted*5)   # 0 is no, 1 or 5 is yes
    # if both are 0 -- 0-0 = 0 TN
    # if gt is 1, pred 5 -- 1-5 = -4 TP
    # if gt is 0, pred 5 -- 0-5 = -5 FP
    # if gt is 1, pred 0 -- 1-0 = 1 FN
    tp = np.where(comb == -4, 1, 0)
    tn = np.where(comb == 0, 1, 0)
    fp = np.where(comb == -5, 1, 0)
    fn = np.where(comb == 1, 1, 0)

    iou = (np.sum(tp))/((np.sum(tp))+np.sum(fp)+np.sum(fn))

    return iou


def IoU(gts, preds):
    iou_per_class = []

    for i in range(4):
        iouscores = idv_iou(preds[i], gts[i])
        # Store the Dice scores per patient
        iou_per_class.append(iouscores)

    return iou_per_class #dice_scores_per_patient








def calculate(gt_np, predicted_np):


    #per class
    gt1 = np.where(gt_np == 1, gt_np, 0)
    gt2 = np.where(gt_np == 2, gt_np, 0)
    gt3 = np.where(gt_np == 3, gt_np, 0)
    gt4 = np.where(gt_np == 4, gt_np, 0)
    #one hot encoding
    onegt1 = np.where(gt1 != 0, 1, 0)
    onegt2 = np.where(gt2 != 0, 1, 0)
    onegt3 = np.where(gt3 != 0, 1, 0)
    onegt4 = np.where(gt4 != 0, 1, 0)

    #per class
    predicted_np1 = np.where(predicted_np == 1, predicted_np, 0)
    predicted_np2 = np.where(predicted_np == 2, predicted_np, 0)
    predicted_np3 = np.where(predicted_np == 3, predicted_np, 0)
    predicted_np4 = np.where(predicted_np == 4, predicted_np, 0)
    #one hot encoding
    onepr1 = np.where(predicted_np1 != 0, 1, 0)
    onepr2 = np.where(predicted_np2 != 0, 1, 0)
    onepr3 = np.where(predicted_np3 != 0, 1, 0)
    onepr4 = np.where(predicted_np4 != 0, 1, 0)


    onegts = [onegt1, onegt2, onegt3, onegt4]
    onepred = [onepr1, onepr2,onepr3, onepr4]



    #clDICE
    cldice = cldice2(onepred, onegts)
    """
    print()
    print('cldice')
    print(cldice)
    """


    #DICE
    dice = volume_dice(onepred, onegts)
    """
    print()
    print('dice')
    print(dice)
    """

    #IoU*
    iou = IoU(onegts, onepred)
    #print()
    #print('IoU', iou)

    #Distance
    patient_hd, patient_95hd, patient_asd = distance_metric(onegts, onepred)
    """
    print()
    print('patient_hd', patient_hd)
    print()
    print('patient_95hd',patient_95hd)
    print()
    print('patient_asd',patient_asd)
    """

    return [dice, cldice, iou, patient_hd, patient_95hd, patient_asd]


   


"""
predicted = nib.load('validation/SEGT_004.nii.gz')  #loaad reference (correct) .nii.gz
predicted_np = np.array(predicted.dataobj)    # to np arraay
pred = torch.from_numpy(predicted_np)

gt_ = nib.load('data/segthor_train/train/Patient_04/GT.nii.gz')  #loaad reference (correct) .nii.gz
gt_np = np.array(gt_.dataobj)    # to np arraay
gt = torch.from_numpy(gt_np)

calculate(gt_np, predicted_np)
"""

"""
x = np.array([1,2,3])
y = np.array([4,5,6])

both = np.array([x,y])
print('MEANNNNNN')
mean = np.mean(both, axis=0)

print(both)
print(mean)
"""


id_list = [3,4,6,20,22,26,29,33]
patient_results = []

dice_ = []
cldice_ = []
iou_ = []
hd = []
hd95 = []
asd = []

print_counter = 0
for patient_id in id_list:
    print_counter+=1
    print('processing',print_counter,'/',len(id_list))

    file_pred = f'validation/SEGT_0{patient_id:02}.nii.gz'
    predicted = nib.load(file_pred)  #loaad reference (correct) .nii.gz
    predicted_np = np.array(predicted.dataobj)    # to np arraay

    file_gt = f'data/segthor_train/train/Patient_{patient_id:02}/GT.nii.gz'
    gt_ = nib.load(file_gt)  #loaad reference (correct) .nii.gz
    gt_np = np.array(gt_.dataobj)    # to np arraay


    patient = calculate(gt_np, predicted_np)
    patient_results.append(patient)
    dice_.append(patient[0])
    cldice_.append(patient[1])
    iou_.append(patient[2])
    hd.append(patient[3])
    hd95.append(patient[4])
    asd.append(patient[5])

print('\n','\n')
print('DICE:')
print(np.mean(dice_, axis=0), '\n')
print('clDICE:')
print(np.mean(cldice_, axis=0), '\n')
print('IoU:')
print(np.mean(iou_, axis=0), '\n')
print('HD:')
print(np.mean(hd, axis=0), '\n')
print('HD95:')
print(np.mean(hd95, axis=0), '\n')
print('ASD:')
print(np.mean(asd, axis=0), '\n')


class1='Esophagus'
class2='Heart'
class3='Trachea'
class4='Aorta'
print('\n')
#print(tabulate.tabulate(dice_, headers=[class1,class2,class3,class4]))

#np.mean(dice_, axis=0),np.mean(cldice_, axis=0),np.mean(iou_, axis=0),np.mean(hd, axis=0),np.mean(hd95, axis=0),np.mean(asd, axis=0)
d = np.mean(dice_, axis=0)
c = np.mean(cldice_, axis=0)
iu = np.mean(iou_, axis=0)
h = np.mean(hd, axis=0)
h95 = np.mean(hd95, axis=0)
a = np.mean(asd, axis=0)


d = np.append('DICE', d)
c = np.append('clDICE', c)
iu = np.append('IoU', iu)
h = np.append('HD', h)
h95 = np.append('HD95', h95)
a = np.append('ASD', a)


results = np.array([d,c,iu,h,h95,a])
#results = np.insert(results, 0, ['DICE','clDICE', 'IoU', 'HD', 'HD95', 'ASD'], axis=1)

print(tabulate.tabulate(results, headers=['Metric',class1,class2,class3,class4]))





"""
#DICE 

cldice_score = 0.0
# Check if both volumes are empty (no segmentation)
if np.sum(predicted_np) == 0 and np.sum(gt_np) == 0:
    cldice_score = 1.0  # Both empty, perfect match 
# If one volume is empty, use worst score
elif np.sum(predicted_np) == 0 or np.sum(gt_np) == 0:
    cldice_score = 0.0
else:
    tprec = cl_score(gt_np,skeletonize(predicted_np)) # susceptible to false positives 
    tsens = cl_score(predicted_np,skeletonize(gt_np)) # susceptible to false negatives.
    if tprec + tsens == 0:
        cldice_score = 0.0
    else:
        cldice_score = 2 * ((tprec * tsens) / (tprec + tsens))

print('cldice_score', cldice_score)





#---DICE--- 



gt_list = [gt1,gt2,gt3,gt4]
pred_list = [predicted_np1,predicted_np2,predicted_np3,predicted_np4]

print(predicted_np.shape)

H, W, slices= predicted_np.shape
max_diagonal_distance = np.sqrt(slices**2 + H**2 + W**2)

classes = K = 4


patient_hd = []
patient_95hd =[]
patient_asd = []




for class_idx in range(0, K):
    print('class_idx', class_idx)
    #pred_volume = predicted_np[:, class_idx, :, :].numpy()
    #gt_volume = gt_np[:, class_idx, :, :].numpy()

    # Check if both volumes are empty (no segmentation) 
    if np.sum(pred_list[class_idx]) == 0 and np.sum(gt_list[class_idx]) == 0:
        hausdorff_distance_class = 0.0
        _95hausdorff_distance_class = 0.0
        asd_class = 0.0
    # If one volume is empty, use the max diagonal distance
    elif np.sum(pred_list[class_idx]) == 0 or np.sum(gt_list[class_idx]) == 0:
        hausdorff_distance_class = max_diagonal_distance
        _95hausdorff_distance_class = max_diagonal_distance
        asd_class = max_diagonal_distance
    else:
        metrics = sg.write_metrics(
            labels=[1],  # Because we have binary volumes
            gdth_img=gt_list[class_idx],
            pred_img=pred_list[class_idx],
            metrics=['hd', 'hd95', 'msd'] 
        )
        hausdorff_distance_class = metrics[0]['hd'][0] 
        _95hausdorff_distance_class = metrics[0]['hd95'][0]
        asd_class = metrics[0]['msd'][0]

    patient_asd.append(asd_class)
    patient_hd.append(hausdorff_distance_class)
    patient_95hd.append(_95hausdorff_distance_class)



print( 'patient_hd', patient_hd)

print( 'patient_95hd', patient_95hd)

print( 'patient_asd', patient_asd)

"""

