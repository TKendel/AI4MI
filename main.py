



# #!/usr/bin/env python3

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

import argparse
import warnings
from typing import Any
from pathlib import Path
from pprint import pprint
from operator import itemgetter
from shutil import copytree, rmtree
import os 

import torch
import numpy as np
import torch.nn.functional as F
from torch import nn, Tensor
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler # learning rate scheduler 

from dataset import SliceDataset
from ShallowNet import shallowCNN
from ENet import ENet
from utils import (Dcm,
                   class2one_hot,
                   probs2one_hot,
                   probs2class,
                   tqdm_,
                   dice_coef,
                   iou_coef,
                   save_images)

from metrics import volume_dice, volume_iou, distance_based_metrics, cldice # volume_hausdorff, slice_hausdorff, avg_surface_distance
from losses import CrossEntropy, DiceLoss, BinaryFocalLoss, GeneralizedDice, CombinedLoss



datasets_params: dict[str, dict[str, Any]] = {}
# K for the number of classes
# Avoids the clases with C (often used for the number of Channel)
datasets_params["TOY2"] = {'K': 2, 'net': shallowCNN, 'B': 2}
datasets_params["SEGTHOR"] = {'K': 5, 'net': ENet, 'B': 8}


# MAIN.PY WITHOUT MODEL SELECTION: 
def setup(args) -> tuple[nn.Module, Any, Any, DataLoader, DataLoader, int]:
    # Networks and scheduler
    gpu: bool = args.gpu and torch.cuda.is_available()
    device = torch.device("cuda") if gpu else torch.device("cpu")
    print(f">> Picked {device} to run experiments")

    K: int = datasets_params[args.dataset]['K']
    net = datasets_params[args.dataset]['net'](1, K)
    net.init_weights()
    net.to(device)

    # Learning rate & adam optimizer
    lr = 0.0005
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999))
    # optimizer = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.01)

    # Adding a learning rate scheduler
    #scheduler = lr_scheduler.PolynomialLR(optimizer, total_iters=5, power=1.0)
    #scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
    #scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

    # Dataset part
    B: int = datasets_params[args.dataset]['B']
    root_dir = Path("data") / args.dataset

    img_transform = transforms.Compose([
        lambda img: img.convert('L'),  #converts to grayscale 
        lambda img: np.array(img)[np.newaxis, ...],
        lambda nd: nd / 255,  # max <= 1   # normalises
        lambda nd: torch.tensor(nd, dtype=torch.float32)
    ])

    gt_transform = transforms.Compose([
        lambda img: np.array(img)[...],
        # The idea is that the classes are mapped to {0, 255} for binary cases
        # {0, 85, 170, 255} for 4 classes
        # {0, 51, 102, 153, 204, 255} for 6 classes
        # Very sketchy but that works here and that simplifies visualization
        lambda nd: nd / (255 / (K - 1)) if K != 5 else nd / 63,  # max <= 1
        lambda nd: torch.tensor(nd, dtype=torch.int64)[None, ...],  # Add one dimension to simulate batch
        lambda t: class2one_hot(t, K=K),
        itemgetter(0)
    ])

    train_set = SliceDataset('train',
                             root_dir,
                             img_transform=img_transform,
                             gt_transform=gt_transform,
                             debug=args.debug)
    train_loader = DataLoader(train_set,
                              batch_size=B,
                              num_workers=args.num_workers,
                              shuffle=True)  # use to be True --> but because of the way i implemented the 3d dice evaluation to work, the original order needs to be perserved...

    val_set = SliceDataset('val',
                           root_dir,
                           img_transform=img_transform,
                           gt_transform=gt_transform,
                           debug=args.debug)
    val_loader = DataLoader(val_set,
                            batch_size=B,
                            num_workers=args.num_workers,
                            shuffle=False)

    args.dest.mkdir(parents=True, exist_ok=True)

    return (net, optimizer, device, train_loader, val_loader, K)


def runTraining(args):
    if args.dataset =='SEGTHOR': 
        sampleV = 10
    elif args.dataset =='TOY':
        sampleV = 100
    
    print(f">>> Setting up to train on {args.dataset} with {args.mode}")
    net, optimizer, device, train_loader, val_loader, K = setup(args)

    if args.mode == "full":
        loss_fn = CrossEntropy(idk=list(range(K)))  # Supervise both background and foreground
        # dloss_fn = DiceLoss(idk=list(range(K)))  # Supervise both background and foreground
        gdloss_fn = GeneralizedDice(idk=list(range(K)))
        # closs_fn = CombinedLoss(idk=list(range(K)))
        #fl_loss_fn = BinaryFocalLoss(cross_entropy=loss_fn, idk=list(range(K)))
    # elif args.mode in ["partial"] and args.dataset in ['SEGTHOR', 'SEGTHOR_STUDENTS']:
    #     loss_fn = CrossEntropy(idk=[0, 1, 3, 4])  # Do not supervise the heart (class 2)
    #     dloss_fn = DiceLoss(idk=[0, 1, 3, 4])  # Do not supervise the heart (class 2)
    #     fl_loss_fn = BinaryFocalLoss(cross_entropy=loss_fn, idk=[0, 1, 3, 4]) # Do not supervise the heart (class 2)
    else:
        raise ValueError(args.mode, args.dataset)

    # Notice one has the length of the _loader_, and the other one of the _dataset_
    # Losses
    log_loss_tra: Tensor = torch.zeros((args.epochs, len(train_loader)))  
    # log_dloss_tra: Tensor = torch.zeros((args.epochs, len(train_loader)))  
    log_gdloss_tra: Tensor = torch.zeros((args.epochs, len(train_loader))) 
    #log_closs_tra: Tensor = torch.zeros((args.epochs, len(train_loader)))  
    # log_focal_tra: Tensor = torch.zeros((args.epochs, len(train_loader)))
    # Metrics 2d
    log_dice_tra: Tensor = torch.zeros((args.epochs, len(train_loader.dataset), K)) 
    log_IOU_tra: Tensor = torch.zeros((args.epochs, len(train_loader.dataset), K))

    # Losses
    log_loss_val: Tensor = torch.zeros((args.epochs, len(val_loader)))
    #log_closs_val: Tensor = torch.zeros((args.epochs, len(val_loader)))
    # log_dloss_val: Tensor = torch.zeros((args.epochs, len(val_loader))) 
    log_gdloss_val: Tensor = torch.zeros((args.epochs, len(val_loader)))
    # log_focal_val: Tensor = torch.zeros((args.epochs, len(val_loader)))
    # Metrics 2d
    log_dice_val: Tensor = torch.zeros((args.epochs, len(val_loader.dataset), K))
    log_IOU_val: Tensor = torch.zeros((args.epochs, len(val_loader.dataset), K))

    # Metrics 3d - only for validation
    log_3d_dice_val = torch.zeros((args.epochs, sampleV, K))  
    log_3d_IOU_val = torch.zeros((args.epochs, sampleV, K))  
    log_hausdorff_val = torch.zeros((args.epochs, sampleV, K-1))   # do not calculate hd for background
    log_95hausdorff_val = torch.zeros((args.epochs, sampleV, K-1)) # do not calculate hd for background
    log_asd_val = torch.zeros((args.epochs, sampleV, K-1)) # do not calculate asd for background
    log_cldice_val = torch.zeros((args.epochs, sampleV, 2)) #only for esophagus and aorta


    #initialize the 'best scores'
    best_dice: float = 0
    # best_iou: float = 0
    # best_95hd: float = float('inf')  # Hausdorff is minimized, so initialize to infinity
    # best_3d_dice: float = 0
    
    for e in range(args.epochs):
        for m in ['train', 'val']:
            # Because we cannot get python 3.11 running in Snellius, we changed the match cases to if statements
            if m == 'train':
                net.train()
                opt = optimizer
                cm = Dcm
                desc = f">> Training   ({e: 4d})"
                loader = train_loader
                log_loss = log_loss_tra
                #log_closs = log_closs_tra
                # log_dloss = log_dloss_tra
                log_gdloss = log_gdloss_tra
                # log_focal = log_focal_tra
                log_dice = log_dice_tra
                log_IOU = log_IOU_tra
            if m == 'val':
                net.eval()
                opt = None
                cm = torch.no_grad
                desc = f">> Validation ({e: 4d})"
                loader = val_loader
                log_loss = log_loss_val
                #log_closs = log_closs_val
                # log_focal = log_focal_val
                # log_dloss = log_dloss_val
                log_gdloss = log_gdloss_val
                log_dice = log_dice_val
                log_IOU = log_IOU_val 
                log_3d_dice = log_3d_dice_val
                log_3d_IOU = log_3d_IOU_val
                log_hausdorff = log_hausdorff_val
                log_95hausdorff = log_95hausdorff_val
                #log_slicehd = log_slicehd_val
                log_asd = log_asd_val
                log_cldice = log_cldice_val

                all_predictions = [] # store the predictions each epoch
                all_gt_slices = [] # store the gts each epoch
            #If we ever get python 3.11, we can change to match and remove the upper two if statements
            """
            match m:
                case 'train':
                    net.train()
                    opt = optimizer
                    cm = Dcm
                    desc = f">> Training   ({e: 4d})"
                    loader = train_loader
                    log_loss = log_loss_tra
                    log_dice = log_dice_tra
                case 'val':
                    net.eval()
                    opt = None
                    cm = torch.no_grad
                    desc = f">> Validation ({e: 4d})"
                    loader = val_loader
                    log_loss = log_loss_val
                    log_dice = log_dice_val
              """

            with cm():  # Either dummy context manager, or the torch.no_grad for validation
                j = 0
                tq_iter = tqdm_(enumerate(loader), total=len(loader), desc=desc)
                for i, data in tq_iter:  # i = batch
                    img = data['images'].to(device)
                    gt = data['gts'].to(device)

                    if opt:  # So only for training
                        opt.zero_grad()  #ensures that gradients are not accumulated from previous batches

                    # Sanity tests to see we loaded and encoded the data correctly
                    assert 0 <= img.min() and img.max() <= 1
                    B, _, W, H = img.shape  # _ = nr. channels

                    pred_logits = net(img)
                    pred_probs = F.softmax(1 * pred_logits, dim=1)  # 1 is the temperature parameter

                    
                    
                    # Metrics computation, not used for training
                    pred_seg = probs2one_hot(pred_probs)  # shape (B, C, H, W)
                    # In order to calculate dice on 3d, we collect all predictions resulting from a single forward pass + their corresponding gts
                    if m == 'val':  
                        all_predictions.append(pred_seg.cpu())
                        all_gt_slices.append(gt.cpu())
                 
                    log_dice[e, j:j + B, :] = dice_coef(gt, pred_seg)  # One DSC value per sample and per class
                    log_IOU[e, j:j + B, :] = iou_coef(gt, pred_seg)  # One IOU value per sample and per class


                    # cross entropy loss 
                    loss = loss_fn(pred_probs, gt)
                    log_loss[e, i] = loss.item()  # One loss value per batch (averaged in the loss)

                    # closs = closs_fn(pred_probs, gt)
                    # log_closs[e, i] = closs.item()

                    # # dice loss
                    # dloss = dloss_fn(pred_probs, gt)
                    # log_dloss[e, i] = dloss.item() 

                    gdloss = gdloss_fn(pred_probs, gt)
                    log_gdloss[e, i] = gdloss.item() 

                    # focal loss
                    # floss = fl_loss_fn(pred_probs, gt)
                    # log_focal[e, i] = floss.item()


                    # MAKE SURE TO SPECIFY THE CORRECT LOSS FUNCTION HERE - LOSS(ce loss), DLOSS (dice loss), 
                    # FLOSS (focal loss), GDLOSS (generalised dice loss), CLOSS (combined loss = 0.5*ce + 0.5*dice)
                    if opt:  # Only for training
                        loss.backward()
                        opt.step()

                    if m == 'val':
                        with warnings.catch_warnings():
                            warnings.filterwarnings('ignore', category=UserWarning)
                            predicted_class: Tensor = probs2class(pred_probs)
                            mult: int = 63 if K == 5 else (255 / (K - 1))
                            save_images(predicted_class * mult,
                                        data['stems'],
                                        args.dest / f"iter{e:03d}" / m)

                    j += B  # Keep in mind that _in theory_, each batch might have a different size
                    # For the DSC & IOU average: do not take the background class (0) into account:
                    postfix_dict: dict[str, str] = {"Dice": f"{log_dice[e, :j, 1:].mean():05.3f}",
                                                    "IoU": f"{log_IOU[e, :j, 1:].mean():05.3f}",
                                                    "Loss": f"{log_loss[e, :i + 1].mean():5.2e}",
                                                    # "Focal Loss": f"{log_focal[e, :i + 1].mean():5.2e}",
                                                    # "dLoss": f"{log_dloss[e, :i + 1].mean():5.2e}",
                                                    "gdLoss": f"{log_gdloss[e, :i + 1].mean():5.2e}"
                                                    }
                    # Print the means per organ
                    if K > 2:
                        postfix_dict |= {f"Dice-{k}": f"{log_dice[e, :j, k].mean():05.3f}" for k in range(1, K)}
                        postfix_dict |= {f"IoU-{k}": f"{log_IOU[e, :j, k].mean():05.3f}" for k in range(1, K)}
                    tq_iter.set_postfix(postfix_dict)
            
            if m == 'val':
                all_predictions_tensor = torch.cat(all_predictions, dim=0)
                all_gt_tensor = torch.cat(all_gt_slices, dim=0) 
                path_to_slices = os.path.join("data", "SEGTHOR", "val", "img")
                
                # Calculating the 3d sccores 
                dice_scores_per_patient = volume_dice(all_predictions_tensor, all_gt_tensor, path_to_slices)
                iou_scores_per_patient = volume_iou(all_predictions_tensor, all_gt_tensor, path_to_slices)
                hausdorff_per_patient, _95hausdorf_per_patient, asd_per_patient = distance_based_metrics(all_predictions_tensor, all_gt_tensor, path_to_slices, K)
                # slice_based_hd_per_patient = slice_hausdorff(all_predictions_tensor, all_gt_tensor, path_to_slices,K)
                cldice_per_patient = cldice(all_predictions_tensor, all_gt_tensor, path_to_slices)

                assert (dice_scores_per_patient.keys() == iou_scores_per_patient.keys() == hausdorff_per_patient.keys() == _95hausdorf_per_patient.keys() == asd_per_patient.keys() == cldice_per_patient.keys()), "Mismatch in patient keys across different metric dictionaries"
                for patient_idx, patient in enumerate(dice_scores_per_patient.keys()):
                    dice_scores = dice_scores_per_patient[patient]
                    iou_score = iou_scores_per_patient[patient]
                    hausdorff = hausdorff_per_patient[patient]
                    _95hausdorff = _95hausdorf_per_patient[patient]
                    asd = asd_per_patient[patient]
                    # sb_hd = slice_based_hd_per_patient[patient]
                    cldice_score = cldice_per_patient[patient]

                    # Store the metrics in the corresponding log tensors
                    log_3d_dice[e, patient_idx, :] = dice_scores.to(dtype=log_3d_dice.dtype, device=log_3d_dice.device)
                    log_3d_IOU[e, patient_idx, :] = iou_score.to(dtype=log_3d_IOU.dtype, device=log_3d_IOU.device)
                    log_hausdorff[e, patient_idx, :] = hausdorff.to(dtype=log_hausdorff.dtype, device=log_hausdorff.device)
                    log_95hausdorff[e, patient_idx, :] = _95hausdorff.to(dtype=log_95hausdorff.dtype, device=log_95hausdorff.device)
                    log_asd[e, patient_idx, :] = asd.to(dtype=log_asd.dtype, device=log_asd.device)
                    # log_slicehd[e, patient_idx, :] = sb_hd.to(dtype=log_slicehd.dtype, device=log_slicehd.device)  # 
                    log_cldice[e, patient_idx, :] = cldice_score.to(dtype=log_cldice.dtype, device=log_cldice.device)

                
                # Print the metrics - mean (excluding the background) - per organ 
                for metric_name, log_metric in [("3dDice", log_3d_dice), ("3dIOU", log_3d_IOU)]:  
                    print(f"{metric_name}: {log_metric[e, :, 1:].mean():05.3f}\t", end='')  
                    if K > 2:
                        for k in range(1, K):
                            print(f"{metric_name}-{k}: {log_metric[e, :, k].mean():05.3f}\t", end='')   
                    print()
            
                for metric_name, log_metric in [("HD", log_hausdorff), ("95HD", log_95hausdorff), ("ASD", log_asd)]: # ,("slHD", log_slicehd)
                    print(f"{metric_name}: {log_metric[e, :, :].mean():05.3f}\t", end='')  
                    if K > 2:
                        for k in range(0, 4):
                            print(f"{metric_name}-{k+1}: {log_metric[e, :, k].mean():05.3f}\t", end='')  
                    print()
                
                class_indices = {0: 1, 1: 4}
                for metric_name, log_metric in [("clDice", log_cldice)]:
                    print(f"{metric_name}: {log_metric[e, :, :].mean():05.3f}\t", end='')  
                    for k in range(2):
                        class_idx = class_indices[k]  
                        print(f"{metric_name}-{class_idx}: {log_metric[e, :, k].mean():05.3f}\t", end='')
                    print() 

        # I save it at each epoch, in case the code crashes or I decide to stop it early
        np.save(args.dest / "loss_tra.npy", log_loss_tra)
        # np.save(args.dest / "closs_tra.npy", log_closs_tra)
        # np.save(args.dest / "dloss_tra.npy", log_dloss_tra)
        np.save(args.dest / "gdloss_tra.npy", log_gdloss_tra)
        # np.save(args.dest / "floss_tra.npy", log_focal_tra)
        np.save(args.dest / "dice_tra.npy", log_dice_tra)
        np.save(args.dest / "iou_tra.npy", log_IOU_tra)
        
        np.save(args.dest / "loss_val.npy", log_loss_val)
        # np.save(args.dest / "closs_val.npy", log_closs_val)
        # np.save(args.dest / "dloss_val.npy", log_dloss_val)
        np.save(args.dest / "gdloss_val.npy", log_gdloss_val)
        # np.save(args.dest / "floss_val.npy", log_focal_val)
        np.save(args.dest / "dice_val.npy", log_dice_val)
        np.save(args.dest / "iou_val.npy", log_IOU_val)

        np.save(args.dest / "3ddice_val.npy", log_3d_dice_val)
        np.save(args.dest / "3dIOU_val.npy", log_3d_IOU_val)
        #np.save(args.dest / "slHD.npy", log_slicehd)
        np.save(args.dest / "HD_val.npy", log_hausdorff)
        np.save(args.dest / "95HD_val.npy", log_95hausdorff)
        np.save(args.dest / "ASD_val.npy", log_asd_val)
        np.save(args.dest / "cldice_val.npy", log_cldice)

        

        current_dice: float = log_dice_val[e, :, 1:].mean().item()
        # current_iou: float = log_3d_IOU_val[e, :, 1:].mean().item()
        # current_3d_dice: float = log_3d_dice_val[e, :, 1:].mean().item()
        # current_95hd: float = log_95hausdorff[e, :, 1:].mean().item()

        # Check for improvements
        if current_dice > best_dice: # and (current_3d_dice > best_3d_dice) and (current_iou > best_iou) and (current_95hd < best_95hd):
            print(f">>> Improved metrics at epoch {e}:")
            print(f"    Dice: {best_dice:05.3f} -> {current_dice:05.3f} DSC")
            # print(f"    Dice: {best_3d_dice:05.3f} -> {current_3d_dice:05.3f} DSC")
            # print(f"    IoU: {best_iou:05.3f} -> {current_iou:05.3f} IoU")
            # print(f"    Hausdorff: {best_95hd:05.3f} -> {current_95hd:05.3f} HD")

            # Update best metrics
            best_dice = current_dice
            # best_3d_dice = current_3d_dice
            # best_iou = current_iou
            # best_95hd = current_95hd

            # Write the best epoch number to a file
            with open(args.dest / "best_epoch.txt", 'w') as f:
                f.write(str(e))

            # Handle the directory for the best epoch
            best_folder = args.dest / "best_epoch"
            if best_folder.exists():
                rmtree(best_folder)
            copytree(args.dest / f"iter{e:03d}", Path(best_folder))

            # Save the model and its weights
            torch.save(net, args.dest / "bestmodel.pkl")
            torch.save(net.state_dict(), args.dest / "bestweights.pt")

            best_epoch = e

        #stops if metrics don't improve after 5 epochs above epoch 15
        patience = 25 #how many epochs it needs to wait to decide to stop

        if e >= 15:
            if (e - best_epoch) >= patience:
                print(f"Stopping early at epoch {e} due to no improvement in {patience} epochs after epoch {best_epoch}")
                break


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--dataset', default='TOY2', choices=datasets_params.keys())
    parser.add_argument('--mode', default='full', choices=['partial', 'full'])
    parser.add_argument('--dest', type=Path, required=True,
                        help="Destination directory to save the results (predictions and weights).")

    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--debug', action='store_true',
                        help="Keep only a fraction (10 samples) of the datasets, "
                             "to test the logic around epochs and logging easily.")

    args = parser.parse_args()

    pprint(args)

    runTraining(args)


if __name__ == '__main__':
    main()



#### main.py with model selection


# #!/usr/bin/env python3

# # MIT License

# # Copyright (c) 2024 Hoel Kervadec

# # Permission is hereby granted, free of charge, to any person obtaining a copy
# # of this software and associated documentation files (the "Software"), to deal
# # in the Software without restriction, including without limitation the rights
# # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# # copies of the Software, and to permit persons to whom the Software is
# # furnished to do so, subject to the following conditions:

# # The above copyright notice and this permission notice shall be included in all
# # copies or substantial portions of the Software.

# # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# # SOFTWARE.

# import argparse
# import warnings
# from typing import Any
# from pathlib import Path
# from pprint import pprint
# from operator import itemgetter
# from shutil import copytree, rmtree
# import os 

# import torch
# import numpy as np
# import torch.nn.functional as F
# from torch import nn, Tensor
# from torchvision import transforms
# from torch.utils.data import DataLoader
# import torch.optim.lr_scheduler as lr_scheduler # learning rate scheduler 

# from dataset import SliceDataset
# from ShallowNet import shallowCNN
# from ENet import ENet
# from utils import (Dcm,
#                    class2one_hot,
#                    probs2one_hot,
#                    probs2class,
#                    tqdm_,
#                    dice_coef,
#                    iou_coef,
#                    save_images)

# from metrics import volume_dice, volume_iou, distance_based_metrics, cldice # volume_hausdorff, slice_hausdorff, avg_surface_distance
# from losses import CrossEntropy, DiceLoss, BinaryFocalLoss



# datasets_params: dict[str, dict[str, Any]] = {}
# # K for the number of classes
# # Avoids the clases with C (often used for the number of Channel)
# datasets_params["TOY2"] = {'K': 2, 'net': shallowCNN, 'B': 2}
# datasets_params["SEGTHOR"] = {'K': 5, 'net': ENet, 'B': 8}


# def setup(args) -> tuple[nn.Module, Any, Any, DataLoader, DataLoader, int]:
#     # Networks and scheduler
#     gpu: bool = args.gpu and torch.cuda.is_available()
#     device = torch.device("cuda") if gpu else torch.device("cpu")
#     print(f">> Picked {device} to run experiments")

#     K: int = datasets_params[args.dataset]['K']
#     net = datasets_params[args.dataset]['net'](1, K)
#     net.init_weights()
#     net.to(device)

#     # Learning rate & adam optimizer
#     lr = 0.0005
#     optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999))
#     # optimizer = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.01)

#     # Adding a learning rate scheduler
#     #scheduler = lr_scheduler.PolynomialLR(optimizer, total_iters=5, power=1.0)
#     #scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
#     #scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

#     # Dataset part
#     B: int = datasets_params[args.dataset]['B']
#     root_dir = Path("data") / args.dataset

#     img_transform = transforms.Compose([
#         lambda img: img.convert('L'),  #converts to grayscale 
#         lambda img: np.array(img)[np.newaxis, ...],
#         lambda nd: nd / 255,  # max <= 1   # normalises
#         lambda nd: torch.tensor(nd, dtype=torch.float32)
#     ])

#     gt_transform = transforms.Compose([
#         lambda img: np.array(img)[...],
#         # The idea is that the classes are mapped to {0, 255} for binary cases
#         # {0, 85, 170, 255} for 4 classes
#         # {0, 51, 102, 153, 204, 255} for 6 classes
#         # Very sketchy but that works here and that simplifies visualization
#         lambda nd: nd / (255 / (K - 1)) if K != 5 else nd / 63,  # max <= 1
#         lambda nd: torch.tensor(nd, dtype=torch.int64)[None, ...],  # Add one dimension to simulate batch
#         lambda t: class2one_hot(t, K=K),
#         itemgetter(0)
#     ])

#     train_set = SliceDataset('train',
#                              root_dir,
#                              img_transform=img_transform,
#                              gt_transform=gt_transform,
#                              debug=args.debug)
#     train_loader = DataLoader(train_set,
#                               batch_size=B,
#                               num_workers=args.num_workers,
#                               shuffle=True)  # use to be True --> but because of the way i implemented the 3d dice evaluation to work, the original order needs to be perserved...

#     val_set = SliceDataset('val',
#                            root_dir,
#                            img_transform=img_transform,
#                            gt_transform=gt_transform,
#                            debug=args.debug)
#     val_loader = DataLoader(val_set,
#                             batch_size=B,
#                             num_workers=args.num_workers,
#                             shuffle=False)

#     args.dest.mkdir(parents=True, exist_ok=True)

#     return (net, optimizer, device, train_loader, val_loader, K)

# # def normalise(metric, smaller: boolean):
# #     for m in metric:
# #         if smaller:
# #             #add cdf


# def runTraining(args):
#     if args.dataset =='SEGTHOR': 
#         sampleV = 10
#     elif args.dataset =='TOY':
#         sampleV = 100
    
#     print(f">>> Setting up to train on {args.dataset} with {args.mode}")
#     net, optimizer, device, train_loader, val_loader, K = setup(args)

#     if args.mode == "full":
#         loss_fn = CrossEntropy(idk=list(range(K)))  # Supervise both background and foreground
#         dloss_fn = DiceLoss(idk=list(range(K)))  # Supervise both background and foreground
#         fl_loss_fn = BinaryFocalLoss(cross_entropy=loss_fn, idk=list(range(K)))
#     elif args.mode in ["partial"] and args.dataset in ['SEGTHOR', 'SEGTHOR_STUDENTS']:
#         loss_fn = CrossEntropy(idk=[0, 1, 3, 4])  # Do not supervise the heart (class 2)
#         dloss_fn = DiceLoss(idk=[0, 1, 3, 4])  # Do not supervise the heart (class 2)
#         fl_loss_fn = BinaryFocalLoss(cross_entropy=loss_fn, idk=[0, 1, 3, 4]) # Do not supervise the heart (class 2)
#     else:
#         raise ValueError(args.mode, args.dataset)

#     # Notice one has the length of the _loader_, and the other one of the _dataset_
#     # Losses
#     log_loss_tra: Tensor = torch.zeros((args.epochs, len(train_loader)))  
#     log_dloss_tra: Tensor = torch.zeros((args.epochs, len(train_loader)))  
#     log_focal_tra: Tensor = torch.zeros((args.epochs, len(train_loader)))
#     # Metrics 2d
#     log_dice_tra: Tensor = torch.zeros((args.epochs, len(train_loader.dataset), K)) 
#     log_IOU_tra: Tensor = torch.zeros((args.epochs, len(train_loader.dataset), K))

#     # Losses
#     log_loss_val: Tensor = torch.zeros((args.epochs, len(val_loader)))
#     log_dloss_val: Tensor = torch.zeros((args.epochs, len(val_loader))) 
#     log_focal_val: Tensor = torch.zeros((args.epochs, len(val_loader)))
#     # Metrics 2d
#     log_dice_val: Tensor = torch.zeros((args.epochs, len(val_loader.dataset), K))
#     log_IOU_val: Tensor = torch.zeros((args.epochs, len(val_loader.dataset), K))

#     # Metrics 3d - only for validation
#     log_3d_dice_val = torch.zeros((args.epochs, sampleV, K))  
#     log_3d_IOU_val = torch.zeros((args.epochs, sampleV, K))  
#     log_hausdorff_val = torch.zeros((args.epochs, sampleV, K-1))   # do not calculate hd for background
#     log_95hausdorff_val = torch.zeros((args.epochs, sampleV, K-1)) # do not calculate hd for background
#     # log_slicehd_val = torch.zeros((args.epochs, sampleV, K-1)) 
#     log_asd_val = torch.zeros((args.epochs, sampleV, K-1)) # do not calculate asd for background
#     log_cldice_val = torch.zeros((args.epochs, sampleV, 2)) #only for esophagus and aorta


#     #initialize the 'best scores'
#     best_dice: float = 0
#     best_iou: float = 0
#     best_95hd: float = float('inf')  # Hausdorff is minimized, so initialize to infinity
#     best_3d_dice: float = 0
    
#     for e in range(args.epochs):
#         for m in ['train', 'val']:
#             # Because we cannot get python 3.11 running in Snellius, we changed the match cases to if statements
#             if m == 'train':
#                 net.train()
#                 opt = optimizer
#                 cm = Dcm
#                 desc = f">> Training   ({e: 4d})"
#                 loader = train_loader
#                 log_loss = log_loss_tra
#                 log_dloss = log_dloss_tra
#                 log_focal = log_focal_tra
#                 log_dice = log_dice_tra
#                 log_IOU = log_IOU_tra
#             if m == 'val':
#                 net.eval()
#                 opt = None
#                 cm = torch.no_grad
#                 desc = f">> Validation ({e: 4d})"
#                 loader = val_loader
#                 log_loss = log_loss_val
#                 log_focal = log_focal_val
#                 log_dloss = log_dloss_val
#                 log_dice = log_dice_val
#                 log_IOU = log_IOU_val 
#                 log_3d_dice = log_3d_dice_val
#                 log_3d_IOU = log_3d_IOU_val
#                 log_hausdorff = log_hausdorff_val
#                 log_95hausdorff = log_95hausdorff_val
#                 #log_slicehd = log_slicehd_val
#                 log_asd = log_asd_val
#                 log_cldice = log_cldice_val

#                 all_predictions = [] # store the predictions each epoch
#                 all_gt_slices = [] # store the gts each epoch
#             #If we ever get python 3.11, we can change to match and remove the upper two if statements
#             """
#             match m:
#                 case 'train':
#                     net.train()
#                     opt = optimizer
#                     cm = Dcm
#                     desc = f">> Training   ({e: 4d})"
#                     loader = train_loader
#                     log_loss = log_loss_tra
#                     log_dice = log_dice_tra
#                 case 'val':
#                     net.eval()
#                     opt = None
#                     cm = torch.no_grad
#                     desc = f">> Validation ({e: 4d})"
#                     loader = val_loader
#                     log_loss = log_loss_val
#                     log_dice = log_dice_val
#               """

#             with cm():  # Either dummy context manager, or the torch.no_grad for validation
#                 j = 0
#                 tq_iter = tqdm_(enumerate(loader), total=len(loader), desc=desc)
#                 for i, data in tq_iter:  # i = batch
#                     img = data['images'].to(device)
#                     gt = data['gts'].to(device)

#                     if opt:  # So only for training
#                         opt.zero_grad()  #ensures that gradients are not accumulated from previous batches

#                     # Sanity tests to see we loaded and encoded the data correctly
#                     assert 0 <= img.min() and img.max() <= 1
#                     B, _, W, H = img.shape  # _ = nr. channels

#                     pred_logits = net(img)
#                     pred_probs = F.softmax(1 * pred_logits, dim=1)  # 1 is the temperature parameter

                    
                    
#                     # Metrics computation, not used for training
#                     pred_seg = probs2one_hot(pred_probs)  # shape (B, C, H, W)
#                     # In order to calculate dice on 3d, we collect all predictions resulting from a single forward pass + their corresponding gts
#                     if m == 'val':  
#                         all_predictions.append(pred_seg.cpu())
#                         all_gt_slices.append(gt.cpu())
                 
#                     log_dice[e, j:j + B, :] = dice_coef(gt, pred_seg)  # One DSC value per sample and per class
#                     log_IOU[e, j:j + B, :] = iou_coef(gt, pred_seg)  # One IOU value per sample and per class


#                     # cross entropy loss 
#                     loss = loss_fn(pred_probs, gt)
#                     log_loss[e, i] = loss.item()  # One loss value per batch (averaged in the loss)

#                     # dice loss
#                     dloss = dloss_fn(pred_probs, gt)
#                     log_dloss[e, i] = dloss.item() 

#                     # focal loss
#                     floss = fl_loss_fn(pred_probs, gt)
#                     log_focal[e, i] = floss.item()


#                     # MAKE SURE TO SPECIFY THE CORRECT LOSS FUNCTION HERE - LOSS, DLOSS, FLOSS
#                     if opt:  # Only for training
#                         loss.backward()
#                         opt.step()

#                     if m == 'val':
#                         with warnings.catch_warnings():
#                             warnings.filterwarnings('ignore', category=UserWarning)
#                             predicted_class: Tensor = probs2class(pred_probs)
#                             mult: int = 63 if K == 5 else (255 / (K - 1))
#                             save_images(predicted_class * mult,
#                                         data['stems'],
#                                         args.dest / f"iter{e:03d}" / m)

#                     j += B  # Keep in mind that _in theory_, each batch might have a different size
#                     # For the DSC & IOU average: do not take the background class (0) into account:
#                     postfix_dict: dict[str, str] = {"Dice": f"{log_dice[e, :j, 1:].mean():05.3f}",
#                                                     "IoU": f"{log_IOU[e, :j, 1:].mean():05.3f}",
#                                                     "Loss": f"{log_loss[e, :i + 1].mean():5.2e}",
#                                                     "Focal Loss": f"{log_focal[e, :i + 1].mean():5.2e}",
#                                                     "dLoss": f"{log_dloss[e, :i + 1].mean():5.2e}"
#                                                     }
#                     # Print the means per organ
#                     if K > 2:
#                         postfix_dict |= {f"Dice-{k}": f"{log_dice[e, :j, k].mean():05.3f}" for k in range(1, K)}
#                         postfix_dict |= {f"IoU-{k}": f"{log_IOU[e, :j, k].mean():05.3f}" for k in range(1, K)}
#                     tq_iter.set_postfix(postfix_dict)
            
#             if m == 'val':
#                 all_predictions_tensor = torch.cat(all_predictions, dim=0)
#                 all_gt_tensor = torch.cat(all_gt_slices, dim=0) 
#                 path_to_slices = os.path.join("data", "SEGTHOR", "val", "img")
                
#                 # Calculating the 3d sccores 
#                 dice_scores_per_patient = volume_dice(all_predictions_tensor, all_gt_tensor, path_to_slices)
#                 iou_scores_per_patient = volume_iou(all_predictions_tensor, all_gt_tensor, path_to_slices)
#                 hausdorff_per_patient, _95hausdorf_per_patient, asd_per_patient = distance_based_metrics(all_predictions_tensor, all_gt_tensor, path_to_slices, K)
#                 # slice_based_hd_per_patient = slice_hausdorff(all_predictions_tensor, all_gt_tensor, path_to_slices,K)
#                 cldice_per_patient = cldice(all_predictions_tensor, all_gt_tensor, path_to_slices)

#                 assert (dice_scores_per_patient.keys() == iou_scores_per_patient.keys() == hausdorff_per_patient.keys() == _95hausdorf_per_patient.keys() == asd_per_patient.keys() == cldice_per_patient.keys()), "Mismatch in patient keys across different metric dictionaries"
#                 for patient_idx, patient in enumerate(dice_scores_per_patient.keys()):
#                     dice_scores = dice_scores_per_patient[patient]
#                     iou_score = iou_scores_per_patient[patient]
#                     hausdorff = hausdorff_per_patient[patient]
#                     _95hausdorff = _95hausdorf_per_patient[patient]
#                     asd = asd_per_patient[patient]
#                     # sb_hd = slice_based_hd_per_patient[patient]
#                     cldice_score = cldice_per_patient[patient]

#                     # Store the metrics in the corresponding log tensors
#                     log_3d_dice[e, patient_idx, :] = dice_scores.to(dtype=log_3d_dice.dtype, device=log_3d_dice.device)
#                     log_3d_IOU[e, patient_idx, :] = iou_score.to(dtype=log_3d_IOU.dtype, device=log_3d_IOU.device)
#                     log_hausdorff[e, patient_idx, :] = hausdorff.to(dtype=log_hausdorff.dtype, device=log_hausdorff.device)
#                     log_95hausdorff[e, patient_idx, :] = _95hausdorff.to(dtype=log_95hausdorff.dtype, device=log_95hausdorff.device)
#                     log_asd[e, patient_idx, :] = asd.to(dtype=log_asd.dtype, device=log_asd.device)
#                     # log_slicehd[e, patient_idx, :] = sb_hd.to(dtype=log_slicehd.dtype, device=log_slicehd.device)  # 
#                     log_cldice[e, patient_idx, :] = cldice_score.to(dtype=log_cldice.dtype, device=log_cldice.device)

#                 print(f"log_3d_dice: {log_3d_dice}")
#                 # Print the metrics - mean (excluding the background) - per organ 
#                 for metric_name, log_metric in [("3dDice", log_3d_dice), ("3dIOU", log_3d_IOU)]:  
#                     print(f"{metric_name}: {log_metric[e, :, 1:].mean():05.3f}\t", end='')  
#                     if K > 2:
#                         for k in range(1, K):
#                             print(f"{metric_name}-{k}: {log_metric[e, :, k].mean():05.3f}\t", end='')   
#                     print()
            
#                 for metric_name, log_metric in [("HD", log_hausdorff), ("95HD", log_95hausdorff), ("ASD", log_asd)]: # ,("slHD", log_slicehd)
#                     print(f"{metric_name}: {log_metric[e, :, :].mean():05.3f}\t", end='')  
#                     if K > 2:
#                         for k in range(0, 4):
#                             print(f"{metric_name}-{k+1}: {log_metric[e, :, k].mean():05.3f}\t", end='')  
#                     print()
                
#                 class_indices = {0: 1, 1: 4}
#                 for metric_name, log_metric in [("clDice", log_cldice)]:
#                     print(f"{metric_name}: {log_metric[e, :, :].mean():05.3f}\t", end='')  
#                     for k in range(2):
#                         class_idx = class_indices[k]  
#                         print(f"{metric_name}-{class_idx}: {log_metric[e, :, k].mean():05.3f}\t", end='')
#                     print() 

#         # I save it at each epoch, in case the code crashes or I decide to stop it early
#         np.save(args.dest / "loss_tra.npy", log_loss_tra)
#         np.save(args.dest / "dloss_tra.npy", log_dloss_tra)
#         np.save(args.dest / "floss_tra.npy", log_focal_tra)
#         np.save(args.dest / "dice_tra.npy", log_dice_tra)
#         np.save(args.dest / "iou_tra.npy", log_IOU_tra)
        
#         np.save(args.dest / "loss_val.npy", log_loss_val)
#         np.save(args.dest / "dloss_val.npy", log_dloss_val)
#         np.save(args.dest / "floss_val.npy", log_focal_val)
#         np.save(args.dest / "dice_val.npy", log_dice_val)
#         np.save(args.dest / "iou_val.npy", log_IOU_val)

#         np.save(args.dest / "3ddice_val.npy", log_3d_dice_val)
#         np.save(args.dest / "3dIOU_val.npy", log_3d_IOU_val)
#         #np.save(args.dest / "slHD.npy", log_slicehd)
#         np.save(args.dest / "HD_val.npy", log_hausdorff)
#         np.save(args.dest / "95HD_val.npy", log_95hausdorff)
#         np.save(args.dest / "ASD_val.npy", log_asd_val)
#         np.save(args.dest / "cldice_val.npy", log_cldice)

        

#         current_dice: float = log_dice_val[e, :, 1:].mean().item()
#         current_iou: float = log_3d_IOU_val[e, :, 1:].mean().item()
#         current_3d_dice: float = log_3d_dice_val[e, :, 1:].mean().item()
#         current_95hd: float = log_95hausdorff[e, :, 1:].mean().item()

#         # Check for improvements
#         if (current_dice > best_dice) and (current_3d_dice > best_3d_dice) and (current_iou > best_iou) and (current_95hd < best_95hd):
#             print(f">>> Improved metrics at epoch {e}:")
#             print(f"    Dice: {best_dice:05.3f} -> {current_dice:05.3f} DSC")
#             print(f"    3D_Dice: {best_3d_dice:05.3f} -> {current_3d_dice:05.3f} DSC")
#             print(f"    IoU: {best_iou:05.3f} -> {current_iou:05.3f} IoU")
#             print(f"    Hausdorff: {best_95hd:05.3f} -> {current_95hd:05.3f} HD")

#             # Update best metrics
#             best_dice = current_dice
#             best_3d_dice = current_3d_dice
#             best_iou = current_iou
#             best_95hd = current_95hd

#             # Write the best epoch number to a file
#             with open(args.dest / "best_epoch.txt", 'w') as f:
#                 f.write(str(e))

#             # Handle the directory for the best epoch
#             best_folder = args.dest / "best_epoch"
#             if best_folder.exists():
#                 rmtree(best_folder)
#             copytree(args.dest / f"iter{e:03d}", Path(best_folder))

#             # Save the model and its weights
#             torch.save(net, args.dest / "bestmodel.pkl")
#             torch.save(net.state_dict(), args.dest / "bestweights.pt")

#             best_epoch = e

#         #stops if metrics don't improve after 5 epochs above epoch 15
#         patience = 5 #how many epochs it needs to wait to decide to stop

#         if e >= 15:
#             if (e - best_epoch) >= patience:
#                 print(f"Stopping early at epoch {e} due to no improvement in {patience} epochs after epoch {best_epoch}")
#                 break


# def main():
#     parser = argparse.ArgumentParser()

#     parser.add_argument('--epochs', default=200, type=int)
#     parser.add_argument('--dataset', default='TOY2', choices=datasets_params.keys())
#     parser.add_argument('--mode', default='full', choices=['partial', 'full'])
#     parser.add_argument('--dest', type=Path, required=True,
#                         help="Destination directory to save the results (predictions and weights).")

#     parser.add_argument('--num_workers', type=int, default=0)
#     parser.add_argument('--gpu', action='store_true')
#     parser.add_argument('--debug', action='store_true',
#                         help="Keep only a fraction (10 samples) of the datasets, "
#                              "to test the logic around epochs and logging easily.")

#     args = parser.parse_args()

#     pprint(args)

#     runTraining(args)


# if __name__ == '__main__':
#     main()

