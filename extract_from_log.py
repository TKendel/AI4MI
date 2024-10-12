

#extract training data from logs

import numpy as np
import matplotlib.pyplot as plt

import sys, os

#get named arguments divided by '='
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

file_path = "logs/log500.txt"
if 'file' in arg_dic:
    file_path = arg_dic['file']

print(file_path)

epochs = []
lrs = []
trs = []
vls = []
ious = []
dpcs = []


#https://stackoverflow.com/questions/70429209/how-do-i-use-python-3-to-find-a-certain-text-line-and-copy-the-lines-below-it
text_lines = open(file_path, "r").readlines()
for i, line in enumerate(text_lines):
    #Extract epoch
    if line.find("Epoch")!=-1:  #-1 when missing, reads if Epoch is NOT missing
        if line.find("time")==-1: #-1 when missing, reads if time is missing
            epoch_line = line.split(': Epoch ')
            epoch_num = epoch_line[1].strip()   # .strip() to remove '\n'
            epochs.append(epoch_num)
    #Extract learning rate
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
    if line.find("IoU")!=-1:  #-1 when missing, reads if Epoch is NOT missing
        iou_line = line.split('IoU')
        iou = iou_line[1].strip()
        ious.append(iou)
    if line.find("DICE PER CLASS")!=-1:  #-1 when missing, reads if Epoch is NOT missing
        dpc_line = line.split('DICE PER CLASS')
        dpc = dpc_line[1].strip()
        dpcs.append(dpc)


folder = 'logs_out'
if 'out' in arg_dic:
    folder = arg_dic['out']
 
epochs_np = np.array(epochs, dtype='int')
print('epochs_np length:', len(epochs_np))
np.save(os.path.join(folder, 'epochs.npy'), epochs_np)

#alt_epochs = np.arange(0,1000, dtype='int')
#print('alt_epochs length:', len(alt_epochs))
#np.save('logs_out/epochs.npy', alt_epochs)

lrs_np = np.array(lrs, dtype='float32')
print('lrs_np length:', len(lrs_np))
np.save(os.path.join(folder, 'learning_rate.npy'), epochs_np)

trs_np = np.array(trs,dtype='float32')
print('trs_np length:', len(trs_np))
np.save(os.path.join(folder, 'loss_tra.npy'), epochs_np)

vls_np = np.array(vls,dtype='float32')
print('vls_np length:', len(vls_np))
np.save(os.path.join(folder, 'loss_val.npy'), epochs_np)

ious_np = np.array(ious,dtype='float32')
print('ious_np length:', len(ious_np))
np.save(os.path.join(folder, 'iou.npy'), epochs_np)

dpcs_np = np.array(dpcs,dtype='float32')
print('dpcs_np length:', len(dpcs_np))
np.save(os.path.join(folder, 'dice.npy'), epochs_np)





if 'show' in sys.argv:
    plt.title("Loss")
    plt.plot(epochs_np, trs_np, color="red", label='Training loss')
    plt.plot(epochs_np, vls_np, color="blue", label='Validation loss')
    plt.legend()
    plt.show()







"""
fig = plt.figure()
ax = fig.gca()
ax.set_title('my title')
ax.plot(alt_epochs, trs_np, linewidth=1.5)
ax.plot(alt_epochs, vls_np, linewidth=1.5)
#ax.set_ylim([0.01, 0.00])   #https://stackoverflow.com/questions/3777861/how-to-set-the-axis-limits
#ax.set_ylim([0.0,1]) 
plt.show()
"""

