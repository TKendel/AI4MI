import numpy as np
import matplotlib.pyplot as plt

# Load the numpy arrays from the files
three_d_dice = np.load('./results/segthor/BASELINE/3ddice_val.npy')
cl_dice = np.load('./results/segthor/BASELINE/cldice_val.npy')


cldice_class1_avg = cl_dice[:, :, 0].mean(axis=1)
cldice_class4_avg = cl_dice[:, :, 1].mean(axis=1)

three_d_dice_class1_avg = three_d_dice[:, :, 1].mean(axis=1)
three_d_dice_class4_avg = three_d_dice[:, :, 4].mean(axis=1)

epochs = np.arange(25)
plt.figure(figsize=(10, 6))
plt.plot(epochs, three_d_dice_class1_avg, label='3D Dice - Esophagus',  color='royalblue')
plt.plot(epochs, cldice_class1_avg, label='clDice - Esophagus ', marker='o', color='royalblue')


plt.plot(epochs, three_d_dice_class4_avg, label='3D Dice - Aorta',  color='firebrick')
plt.plot(epochs, cldice_class4_avg, label='clDice - Aorta', marker='o',color='firebrick')

plt.title('Averaged 3D Dice and clDice Values Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Values')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
