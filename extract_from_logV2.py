import numpy as np
import matplotlib.pyplot as plt
import sys, os

'''
Extract training data from logs

This file extracts data from the nnU-Net logs. Important data is printed out from a modified version
of nnU-Net. We understand that this is rudimentary, however, we attempted outputing np files of such 
data during training, among other, but encountered a plethora of errors; and after much trying, decided
that given the time restriction on this project, this would suffice.
'''

# ++CHECK FOR ARGUMENTS++
# Get named arguments divided by '='
arg_dic = {}
for arg in sys.argv:
    var = arg.split("=")
    if len(var)==2:
        arg_dic[var[0]]=var[1]

# Print help if in variables
if '-h' in sys.argv or '-help' in sys.argv:
    print('HELP', '\n', 'To set the log file for extraction, use file=path_to_file. Default is "logs/log500.txt"',
          '\n', 'To show plotting, pass var "show"',
          '\n', 'To set the log out directory for extraction, use out=directory_name. Default is "logs_out"')

# ++IF GIVEN, UTILIZE SPECIFIED FILEPATH++
file_path = "logs/log500.txt"
if 'file' in arg_dic:
    file_path = arg_dic['file']
print('File path:', file_path)

# ++INFO EXTRACTION++
epochs = []
lrs = []
trs = []
vls = []
tps = []
fps = []
fns = []
dices = []

#https://stackoverflow.com/questions/70429209/how-do-i-use-python-3-to-find-a-certain-text-line-and-copy-the-lines-below-it
text_lines = open(file_path, "r").readlines()
for i, line in enumerate(text_lines):
    # Extract epoch
    if line.find("Epoch")!=-1:  #-1 when missing, reads if Epoch is NOT missing
        if line.find("time")==-1: #-1 when missing, reads if time is missing
            epoch_line = line.split(': Epoch ')
            epoch_num = epoch_line[1].strip()   # .strip() to remove '\n'
            epochs.append(epoch_num)

    # Extract learning rate
    if line.find(" Current learning rate: ")!=-1:  #-1 when missing, reads if Epoch is NOT missing
        lr_line = line.split(': Current learning rate: ')
        lr = lr_line[1].strip()
        lrs.append(lr)

    if line.find(": train_loss ")!=-1:  #-1 when missing, reads if Epoch is NOT missing
        train_loss_line = line.split(': train_loss ')
        tr = train_loss_line[1].strip()
        trs.append(tr)

    if line.find(": val_loss ")!=-1:  #-1 when missing, reads if Epoch is NOT missing
        val_loss_line = line.split(': val_loss ')
        vl = val_loss_line[1].strip()
        vls.append(vl)

    if line.find("_tp_ ")!=-1:  #-1 when missing, reads if Epoch is NOT missing
        tpline = line.strip("_tp_ ")
        tpline = tpline.strip("[")
        tpline = tpline.strip("]")  #this is not enough, strangely
        tpline = tpline.replace("]", '')
        tpline = tpline.strip()
        tpline = tpline.strip(".")
        tpline = tpline.replace(".", '')
        while '  ' in tpline:
            tpline = tpline.replace("  ", ' ')  # remove multiple spaces between numbers
        tplist = tpline.split(' ')
        tparray = np.array(tplist,dtype='float32')
        tps.append(tparray)

    if line.find("_fp_ ")!=-1:  #-1 when missing, reads if Epoch is NOT missing
        fpline = line.strip("_fp_ ")
        fpline = fpline.strip("[")
        fpline = fpline.strip("]")  #this is not enough, strangely
        fpline = fpline.replace("]", '')
        fpline = fpline.strip()
        fpline = fpline.strip(".")
        fpline = fpline.replace(".", '')
        while '  ' in fpline:
            fpline = fpline.replace("  ", ' ')  # remove multiple spaces between numbers
        fplist = fpline.split(' ')
        fparray = np.array(fplist,dtype='float32')
        fps.append(fparray)

    if line.find("_fn_ ")!=-1:  #-1 when missing, reads if Epoch is NOT missing
        fnline = line.strip("_fn_ ")
        fnline = fnline.strip("[")
        fnline = fnline.strip("]")  #this is not enough, strangely
        fnline = fnline.replace("]", '')
        fnline = fnline.strip()
        fnline = fnline.strip(".")
        fnline = fnline.replace(".", '')
        while '  ' in fnline:
            fnline = fnline.replace("  ", ' ')  # remove multiple spaces between numbers
        fnlist = fnline.split(' ')
        fnarray = np.array(fnlist,dtype='float32')
        fns.append(fnarray)

    if line.find("dice_per_class_or_region ")!=-1:  #-1 when missing, reads if Epoch is NOT missing
        dice_class = line.strip("dice_per_class_or_region ")
        dice_class = dice_class.strip('np.float32(')
        dice_class = dice_class.replace('np.float32(', '')
        dice_class = dice_class.replace('[', '')
        dice_class = dice_class.replace(']', '')
        dice_class = dice_class.replace(')', '')
        dice_list = dice_class.split(', ')
        dice_array = np.array(dice_list, dtype='float32')
        dices.append(dice_array)


# ++FURTHER CALCULATIONS++
ious = []

# IoU
for i in range(len(tps)):   #for every entry
    ious.append([])
    for j in range(len(tps[i])):   #for every class (0,1,2,3)
        tp = tps[i][j]
        fn = fns[i][j]
        fp = fps[i][j]
        iou = tp / (tp + fn + fp)
        ious[i].append(iou)

dice2 = []
for i in range(len(tps)):   #for every entry
    dice2.append([])
    for j in range(len(tps[i])):   #for every class (0,1,2,3)
        tp = tps[i][j]
        fn = fns[i][j]
        fp = fps[i][j]
        dc = (2*tp) / ((2*tp) + fn + fp)
        dice2[i].append(dc)



# ++CHECK ARG FOR OUTPUT LOCATION++
folder = 'logs_out2'
if 'out' in arg_dic:
    folder = arg_dic['out']
print('Output directory:', folder)


# ++SAVE NP FILES, + QUICK REPORT++
epochs_np = np.array(epochs, dtype='int')
print('epochs_np length:', len(epochs_np))
np.save(os.path.join(folder, 'epochs.npy'), epochs_np)

lrs_np = np.array(lrs, dtype='float32')
print('lrs_np length:', len(lrs_np))
np.save(os.path.join(folder, 'learning_rate.npy'), lrs_np)

trs_np = np.array(trs,dtype='float32')
print('trs_np length:', len(trs_np))
np.save(os.path.join(folder, 'loss_tra.npy'), trs_np)

vls_np = np.array(vls,dtype='float32')
print('vls_np length:', len(vls_np))
np.save(os.path.join(folder, 'loss_val.npy'), vls_np)

ious_np = np.array(ious,dtype='float32')
print('ious_np length:', len(ious_np))
np.save(os.path.join(folder, 'iou.npy'), ious_np)

dpcs_np = np.array(dices,dtype='float32')
print('dpcs_np length:', len(dpcs_np))
np.save(os.path.join(folder, 'dice.npy'), dpcs_np)

avg_ious = np.mean(ious_np, axis=1)
print('dpcs_np length:', len(avg_ious))
np.save(os.path.join(folder, 'iou_avg.npy'), avg_ious)

avg_dices = np.mean(dpcs_np, axis=1)
print('dpcs_np length:', len(avg_dices))
np.save(os.path.join(folder, 'dice_avg.npy'), avg_dices)

print()
best_epoch = np.argmax(avg_dices)
print('Best epoch', best_epoch)
print('On Best Epoch')
print('Dice')
print(dpcs_np[best_epoch])
print('IoU')
print(ious_np[best_epoch])

print('Best Epoch singletons')
print('Avr Dice',avg_dices[best_epoch], 'Training loss',trs_np[best_epoch], 'Validation Loss',vls_np[best_epoch],'Average IoU', avg_ious[best_epoch])


# ++PLOT THE RESULTS IF THE ARG SHOW IS FOUND++
if 'show' in sys.argv:
    # Classes verified on 3D Slicer
    class1 = 'Esophagus'
    class2 = 'Heart'
    class3 = 'Trachea'
    class4 = 'Aorta'

    color1 = 'seagreen'    #wheat
    color2 = 'darkslateblue'        #orchid
    color3 = 'steelblue'      #royalblue, steelblue
    color4 = 'darkorange'        #aquamarine
    
    # Loss
    plt.title("nnU-Net Loss")
    plt.plot(epochs_np, trs_np, color="tomato", label='Training loss')
    plt.plot(epochs_np, vls_np, color="lightskyblue", label='Validation loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.ylim([-1, 0.4])
    plt.grid(alpha=0.3)
    plt.show()

    # Learning Rate
    plt.title("nnU-Net Learning Rate")
    plt.plot(epochs_np, lrs_np, color="mediumseagreen")
    plt.xlabel("Epochs")
    plt.ylabel("Learning rate")
    plt.ylim([0.0, 0.011])
    plt.grid(alpha=0.3)
    plt.show()
    
    # IoU
    plt.title("nnU-Net IoU")
    plt.plot(epochs_np, avg_ious, color="crimson", label='Average IoU', marker='.')
    plt.plot(epochs_np, ious_np[:, 0], color=color1, label=class1, alpha=0.7)
    plt.plot(epochs_np, ious_np[:, 1], color=color2, label=class2, alpha=0.7)
    plt.plot(epochs_np, ious_np[:, 2], color=color3, label=class3, alpha=0.8)
    plt.plot(epochs_np, ious_np[:, 3], color=color4, label=class4, alpha=0.7)
    plt.xlabel("Epochs")
    plt.ylabel("IoU")
    plt.legend()
    plt.ylim([0.0, 1])
    plt.grid(alpha=0.3)
    plt.show()

    # Dice
    plt.title("nnU-Net Dice")
    plt.plot(epochs_np, avg_dices, color="crimson", label='Average Dice', marker='.')
    plt.plot(epochs_np, dpcs_np[:, 0], color=color1, label=class1, alpha=0.7)
    plt.plot(epochs_np, dpcs_np[:, 1], color=color2, label=class2, alpha=0.7)
    plt.plot(epochs_np, dpcs_np[:, 2], color=color3, label=class3, alpha=0.8)
    plt.plot(epochs_np, dpcs_np[:, 3], color=color4, label=class4, alpha=0.7)
    plt.xlabel("Epochs")
    plt.ylabel("Dice")
    plt.legend()
    plt.ylim([0.0, 1])
    plt.grid(alpha=0.3)
    plt.show()
