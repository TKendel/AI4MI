import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import glob



# distance_metrics = ['HD', '95HD', 'ASD']
# cldice_metrics = ['cldice']
# overlap_metrics = ['3ddice', '3dIOU']
# for filepath in glob.iglob('focallLoss\\*'):
#     data = np.load(filepath)
#     data = data.reshape(-1, data.shape[2])
#     if filepath[11:-4] in overlap_metrics:
#         organ_names = ['Background', 'Esophagus', 'Heart', 'Trachea', 'Aorta']
#     elif filepath[11:-4] in distance_metrics :
#         organ_names = ['Esophagus', 'Heart', 'Trachea', 'Aorta']
#     else:
#         organ_names = ['Esophagus', 'Aorta']
#     df = pd.DataFrame(data, columns=organ_names)
#     sns.boxplot(data=df, linewidth=1)
#     plt.title(filepath[11:-4])
#     plt.savefig(f'{filepath[11:-4]}.png')
#     plt.clf()

data = np.load('focallLoss\\floss_val.npy')
print(data.shape)
# data = data.reshape(-1, data.shape[2])
# organ_names = ['Background', 'Esophagus', 'Heart', 'Trachea', 'Aorta']

# df = pd.DataFrame(data, columns=organ_names)
# sns.boxplot(data=df, linewidth=1)
# plt.title('floss_val')
# plt.savefig(f'floss_val.png')
# plt.clf()