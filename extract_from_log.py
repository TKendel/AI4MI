

#extract training data from logs

import numpy as np
import matplotlib.pyplot as plt

epochs = []
lrs = []
trs = []
vls = []

#https://stackoverflow.com/questions/70429209/how-do-i-use-python-3-to-find-a-certain-text-line-and-copy-the-lines-below-it
text_lines = open("logs/log500.txt", "r").readlines()
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


        
epochs_np = np.array(epochs, dtype='int')
print('epochs_np length:', len(epochs_np))

alt_epochs = np.arange(0,1000, dtype='int')
#print(alt_epochs)
print('alt_epochs length:', len(alt_epochs))
np.save('logs_out/epochs.npy', alt_epochs)

lrs_np = np.array(lrs, dtype='float32')
#print(lrs_np)
print('lrs_np length:', len(lrs_np))
np.save('logs_out/learning_rate.npy', lrs_np)
print(lrs_np.dtype)

trs_np = np.array(trs,dtype='float32')
print('epochs_np length:', len(trs_np))
np.save('logs_out/loss_tra.npy', trs_np)

vls_np = np.array(vls,dtype='float32')
print('epochs_np length:', len(vls_np))
np.save('logs_out/loss_val.npy',vls_np)


#plt.plot(trs_np)
#plt.show()




plt.title("Line graph")
#plt.plot(x, y, color="red")
plt.plot(epochs_np, trs_np, color="red")
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

