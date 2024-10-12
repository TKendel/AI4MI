#!/usr/bin/env python3

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
import torch.optim.lr_scheduler as lr_scheduler #learning rate scheduler 

from dataset import SliceDataset
from ShallowNet import shallowCNN
from ENet import ENet
from utils import (Dcm,
                   class2one_hot,
                   probs2one_hot,
                   probs2class,
                   tqdm_,
                   dice_coef,
                   save_images)
from metrics import volume_dice

from losses import CrossEntropy, DiceLoss

from losses import (CrossEntropy)
from losses import (BinaryFocalLoss) # added


datasets_params: dict[str, dict[str, Any]] = {}
# K for the number of classes
# Avoids the clases with C (often used for the number of Channel)
datasets_params["TOY2"] = {'K': 2, 'net': shallowCNN, 'B': 2}
datasets_params["SEGTHOR"] = {'K': 5, 'net': ENet, 'B': 8}




def setup(args) -> tuple[nn.Module, Any, Any, DataLoader, DataLoader, int]:
    # Networks and scheduler
    gpu: bool = args.gpu and torch.cuda.is_available()
    device = torch.device("cuda") if gpu else torch.device("cpu")
    print(f">> Picked {device} to run experiments")

    K: int = datasets_params[args.dataset]['K']
    net = datasets_params[args.dataset]['net'](1, K)
    net.init_weights()
    net.to(device)

   #learning rate & adam optimizer
    lr = 0.0005
    #optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999))
    #optimizer = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.01)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=0.01)

    #adding a learning rate scheduler
    #scheduler = lr_scheduler.PolynomialLR(optimizer, total_iters=5, power=1.0)
    #scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

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
        sampleT = 30
        sampleV = 10
    elif args.dataset =='TOY':
        sampleT = 1000
        sampleV = 100
    
    print(f">>> Setting up to train on {args.dataset} with {args.mode}")
    net, optimizer, device, train_loader, val_loader, K = setup(args)

    if args.mode == "full":
        # loss_fn = CrossEntropy(idk=list(range(K)))  # Supervise both background and foreground
        # Changed to BinaryFocalLoss
        #loss_fn = BinaryFocalLoss(idk=list(range(K))) 
        loss_fn = CrossEntropy(idk=list(range(K)))  # Supervise both background and foreground
        dloss_fn = DiceLoss(idk=list(range(K)))  # Supervise both background and foreground
    elif args.mode in ["partial"] and args.dataset in ['SEGTHOR', 'SEGTHOR_STUDENTS']:
        # loss_fn = CrossEntropy(idk=[0, 1, 3, 4])  # Do not supervise the heart (class 2)
        # Changed to BinaryFocalLoss
        loss_fn = BinaryFocalLoss(idk=[0, 1, 3, 4])
    else:
        raise ValueError(args.mode, args.dataset)

    # Notice one has the length of the _loader_, and the other one of the _dataset_
    log_loss_tra: Tensor = torch.zeros((args.epochs, len(train_loader)))
    log_dice_tra: Tensor = torch.zeros((args.epochs, len(train_loader.dataset), K))
    # Added log_focal_tra
    log_focal_tra: Tensor = torch.zeros((args.epochs, len(train_loader)))
    log_dloss_tra: Tensor = torch.zeros((args.epochs, len(train_loader)))  # To store the loss for each batch in every epoch during training. lwn(train_laoder) = nr of batches

    log_loss_val: Tensor = torch.zeros((args.epochs, len(val_loader)))
    log_dloss_val: Tensor = torch.zeros((args.epochs, len(train_loader)))  # To store the loss for each batch in every epoch during training. lwn(train_laoder) = nr of batches
    log_dice_val: Tensor = torch.zeros((args.epochs, len(val_loader.dataset), K))
    log_focal_val: Tensor = torch.zeros((args.epochs, len(val_loader)))
    log_3d_dice_val = torch.zeros((args.epochs, sampleV, K))  # Shape: (epochs, num_patients, K)

    best_dice: float = 0

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
                log_dloss = log_dloss_tra
                log_dice = log_dice_tra
                # Added for BinaryFocalLoss
                log_focal = log_focal_tra
            if m == 'val':
                net.eval()
                opt = None
                cm = torch.no_grad
                desc = f">> Validation ({e: 4d})"
                loader = val_loader
                log_loss = log_loss_val
                log_dloss = log_dloss_val
                log_dice = log_dice_val
                # Added loss for BinaryFocalLoss
                log_focal = log_focal_val
                log_3d_dice = log_3d_dice_val
                all_predictions = [] # store the predictions each epoch
                all_gt_slices = [] #store the gts each epoch
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
                    pred_seg = probs2one_hot(pred_probs)  #shape (B, C, H, W)
                    
                    # in order to calculate dice on 3d, we collect all predictions resulting from a single forward pass 
                    # + their corresponding gts
                    if m == 'val':  # Ensure this happens only during validation
                        all_predictions.append(pred_seg.cpu())
                        all_gt_slices.append(gt.cpu())
                        
                    log_dice[e, j:j + B, :] = dice_coef(gt, pred_seg)  # One DSC value per sample and per class
                        # e: The current epoch.
                        # j:j + B: This slices the tensor for the current batch, where:
                        # j is the start index for the current batch in the log_dice array.
                        # j + B is the end index for this batch (B is the batch size, typically 8 in this case).
                        # --> log_dice.shape = (num_epochs, num_samples, num_classes)

                    # Compute focal loss
                    loss = loss_fn(pred_probs, gt)
                    log_loss[e, i] = loss.item()  # One loss value per batch (averaged in the loss)
                    # todo focal: compute focal loss
                    # floss = binary_focal_loss(pred_probs, gt)
                    # log_loss[e, i] = floss.item()  # One loss value per batch (averaged in the loss)
                    dloss = dloss_fn(pred_probs, gt)
                    log_dloss[e, i] = dloss.item() 

                    if opt:  # Only for training
                        loss.backward() #todo focal: change to floss
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
                    # For the DSC average: do not take the background class (0) into account:
                    postfix_dict: dict[str, str] = {"Dice": f"{log_dice[e, :j, 1:].mean():05.3f}",
                                                    "Loss": f"{log_loss[e, :i + 1].mean():5.2e}",
                                                    "Focal Loss": f"{log_focal[e, :i + 1].mean():05.2e}", # Adding the focal loss
                                                    "dLoss": f"{log_dloss[e, :i + 1].mean():5.2e}"}
                    if K > 2:
                        postfix_dict |= {f"Dice-{k}": f"{log_dice[e, :j, k].mean():05.3f}"
                                         for k in range(1, K)}
                    tq_iter.set_postfix(postfix_dict)

            if m == 'val':
                all_predictions_tensor = torch.cat(all_predictions, dim=0)
                all_gt_tensor = torch.cat(all_gt_slices, dim=0) 
                path_to_slices = os.path.join("data", "SEGTHOR", "val", "img")
                dice_scores_per_patient = volume_dice(all_predictions_tensor, all_gt_tensor, path_to_slices)
                print(dice_scores_per_patient)
                for patient_idx, (patient, dice_scores) in enumerate(dice_scores_per_patient.items()):
                    rounded_dice_scores = [float(f"{score:05.3f}") for score in dice_scores]
                    rounded_dice_tensor = torch.tensor(rounded_dice_scores, dtype=torch.float32)
                    log_3d_dice[e, patient_idx, :] = rounded_dice_tensor

                print(f"3d Dice Score (averaged over all patients and classes): {log_3d_dice[e, :, 1:].mean():05.3f}")
                if K > 2:
                    for k in range(1, K):
                        print(f"3dDice-{k}: {log_3d_dice[e, :, k].mean():05.3f}")

        
        print(log_3d_dice[e, :i + 1].mean())
        # I save it at each epochs, in case the code crashes or I decide to stop it early
        np.save(args.dest / "loss_tra.npy", log_loss_tra)
        np.save(args.dest / "dloss_tra.npy", log_dloss_tra)
        np.save(args.dest / "dice_tra.npy", log_dice_tra)
        np.save(args.dest / "focal_tra.npy", log_focal_tra)
        np.save(args.dest / "loss_val.npy", log_loss_val)
        np.save(args.dest / "dloss_val.npy", log_dloss_val)
        np.save(args.dest / "dice_val.npy", log_dice_val)
        np.save(args.dest / "focal_val.npy", log_focal_val)

        np.save(args.dest / "3ddice_val.npy", log_3d_dice_val)
        

        current_dice: float = log_dice_val[e, :, 1:].mean().item()
        if current_dice > best_dice:
            print(f">>> Improved dice at epoch {e}: {best_dice:05.3f}->{current_dice:05.3f} DSC")
            best_dice = current_dice
            with open(args.dest / "best_epoch.txt", 'w') as f:
                    f.write(str(e))

            best_folder = args.dest / "best_epoch"
            if best_folder.exists():
                    rmtree(best_folder)
            copytree(args.dest / f"iter{e:03d}", Path(best_folder))

            torch.save(net, args.dest / "bestmodel.pkl")
            torch.save(net.state_dict(), args.dest / "bestweights.pt")


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
