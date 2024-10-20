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



import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 






def cl_score(v, s):
    return np.sum(v * s) / np.sum(s)

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

"""
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
"""

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
average = results.astype('<U32')
average = np.mean(average, axis=1)
print(average.shape)

print('Mean')
print(tabulate.tabulate(average, headers=['Metric',class1,class2,class3,class4]))
"""
