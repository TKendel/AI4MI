import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os 

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def create_distr_plot(data, organ_names, yliml, ylimu, plot, title, ylabel, colormap, skip_first=False, save_path=None):
    '''
    Creates a distribution plot for the given data depending on the specified plot type ('violin', 'box', or 'swarm').
    '''
    dice_scores = data.reshape(-1, data.shape[2])  # Reshape to have all dice scores per class
    df = pd.DataFrame(dice_scores, columns=organ_names)
    if skip_first:
        df = df.iloc[:, 1:]
    plt.figure(figsize=(8, 6))
    
    if plot=='violin': 
        ax = sns.violinplot(data=df, inner="point", scale="width", palette=colormap, linewidth=1)
        for violin in ax.collections:
            violin.set_facecolor(violin.get_facecolor())  
            violin.set_alpha(0.5)  
    elif plot=='swarm': 
        ax = sns.swarmplot(data=df, palette=colormap, linewidth=0, size=3)
    elif plot =="box": 
         sns.boxplot(data=df, palette=colormap, linewidth=1)
    
    plt.title(title)
    plt.ylabel(ylabel)    
    plt.ylim(yliml, ylimu)
    plt.grid(True, linestyle='--', alpha=0.5)  

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def create_lineplot(data, organ_names, yliml, ylimu, title, ylabel, colormap, skip_first=False, save_path=None):
    '''
    Creates a line plot where metrics are plotted against the number of epochs
    '''
    epochs, slices_or_patient, classes = data.shape
    if skip_first:
        organ_names = organ_names[1:]  # Adjust organ names to exclude the first one
        data = data[:, :, 1:]  # Remove the first class from the data

    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    for i, organ in enumerate(organ_names):
        organ_data_mean = data[:, :, i].mean(axis=1)  # Mean per epoch for this organ
        ax.plot(np.arange(epochs), organ_data_mean, label=organ, color=colormap[organ], linewidth=1.5)

    overall_mean = data.mean(axis=1).mean(axis=1)
    ax.plot(np.arange(epochs), overall_mean, label="All classes", linewidth=2, color='purple')

    ax.legend()
    plt.title(title)
    plt.ylabel(ylabel)
    plt.ylim(yliml, ylimu)
    plt.grid(True, linestyle='--', alpha=0.5)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()


def create_comparison_violin_plot(data1, data2, organ_names, yliml, ylimu, title, ylabel, skip_first=False, save_path=None):
    '''
    Creates a comparison violin plot for two datasets (e.g., CE loss vs Dice loss) across the specified organ classes.
    '''
    dice_scores1 = data1.reshape(-1, data1.shape[2])
    dice_scores2 = data2.reshape(-1, data2.shape[2])

    df1 = pd.DataFrame(dice_scores1, columns=organ_names)
    df2 = pd.DataFrame(dice_scores2, columns=organ_names)

    if skip_first:
        df1 = df1.iloc[:, 1:]
        df2 = df2.iloc[:, 1:]
        organ_names = organ_names[1:]

    # Adjsut these names to what you are comparing
    df1 = df1.melt(var_name='Organ', value_name='Score')
    df1['Dataset'] = 'CE loss'
    df2 = df2.melt(var_name='Organ', value_name='Score')
    df2['Dataset'] = 'Dice loss'

    df_combined = pd.concat([df1, df2])

    plt.figure(figsize=(8, 6))
    sns.violinplot(x='Organ', y='Score', hue='Dataset', data=df_combined, split=True, palette="Set2", inner="point")

    plt.title(title)
    plt.ylabel(ylabel)
    plt.ylim(yliml, ylimu)
    plt.grid(True, linestyle='--', alpha=0.5)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()


def generate_plots(metric_list, organ_names, yliml, ylimu, title_map, ylabel_map, basedir, paperplots_dir, colormap, skip_first=False):
    '''    
    Generates and saves distribution plots (violin, swarm, and box) and a line plot for each metric in the metric list. 
    '''
    for metric in metric_list:
        path = os.path.join(basedir, metric + '_val.npy')  # Correct path construction
        data = np.load(path) 

        # Violin plot & boxplots
        save_boxplot = os.path.join(paperplots_dir, f"{metric}_boxplot.png")
        save_violin = os.path.join(paperplots_dir, f"{metric}_violinplot.png")
        save_swarm = os.path.join(paperplots_dir, f"{metric}_swarmplot.png")

        create_distr_plot(data, organ_names, yliml, ylimu, plot='violin', title=title_map[metric], ylabel=ylabel_map[metric], colormap=colormap, skip_first=skip_first, save_path=save_violin)
        create_distr_plot(data, organ_names, yliml, ylimu,  plot='swarm',title=title_map[metric], ylabel=ylabel_map[metric], colormap=colormap, skip_first=skip_first, save_path=save_swarm)
        create_distr_plot(data, organ_names, yliml, ylimu,  plot='box',title=title_map[metric], ylabel=ylabel_map[metric], colormap=colormap, skip_first=skip_first, save_path=save_boxplot)

        #lineplot
        save_lineplot = os.path.join(paperplots_dir, f"{metric}_lineplot_m.png")
        create_lineplot(data, organ_names, yliml, ylimu, title=title_map[metric], ylabel=ylabel_map[metric], colormap=colormap, skip_first=skip_first, save_path=save_lineplot)
        break


def generate_comparison_plots(metric_list, organ_names, yliml, ylimu, title_map, ylabel_map, basedir, basedir2, paperplots_dir, colormap, skip_first=False):
    ''''
    Generates and saves comparison violin plots for two datasets across the metrics in the metric list.
    '''
    for metric in metric_list:
        path1 = os.path.join(basedir, metric + '_val.npy')
        path2 = os.path.join(basedir2, metric + '_val.npy')

        data1 = np.load(path1)
        data2 = np.load(path2)

        # Violin plot comparison
        save_comparison_violin = os.path.join(paperplots_dir, f"{metric}_comparison_violinplot.png")
        create_comparison_violin_plot(data1, data2, organ_names, yliml, ylimu, title=title_map[metric], ylabel=ylabel_map[metric],skip_first=skip_first, save_path=save_comparison_violin)


################### DATA ###################
basedir = './results/segthor/BASELINE'
basedir2 = './results/segthor/DICELOSS_1510'

color_map = {'Background': 'grey', 'Esophagus': 'dodgerblue', 'Heart': 'orange', 'Trachea': 'yellowgreen', 'Aorta': 'red'}
paperplots_dir = os.path.join('paperplots', os.path.basename(basedir))
os.makedirs(paperplots_dir, exist_ok=True)  

########## Overlap based plots ##########
organ_names = ['Background', 'Esophagus', 'Heart', 'Trachea', 'Aorta']
overlap_metrics = ['3ddice', '3dIOU']
titlemap_overlap = {'3ddice': 'Volumetric Dice - Validation 25 Epochs', '3dIOU': 'Volumetric IoU - Validation 25 Epochs'}
ylabelmap_overlap = {'3ddice': 'Dice', '3dIOU': 'IoU'}
generate_plots(overlap_metrics, organ_names, 0, 1, titlemap_overlap, ylabelmap_overlap, basedir, paperplots_dir, color_map, skip_first=True)
# If you want to have the split violin compare two models;
generate_comparison_plots(overlap_metrics, organ_names, 0, 1, titlemap_overlap, ylabelmap_overlap, basedir, basedir2, paperplots_dir, color_map, skip_first=True)

########## Distance based plots ##########
organ_names_d = ['Esophagus', 'Heart', 'Trachea', 'Aorta']
distance_metrics = ['HD', '95HD', 'ASD']
titlemap_distance = {'HD': 'Hausdorff Distance - Validation 25 Epochs', '95HD': '95 Percentile Hausdorff - Validation 25 Epochs', 'ASD': 'Average Surface Distance - Validation 25 Epochs'}
ylabelmap_distance = {'HD': 'HD', '95HD': '95HD', 'ASD': 'ASD'}
generate_plots(distance_metrics, organ_names_d, 0, 500, titlemap_distance, ylabelmap_distance, basedir, paperplots_dir, color_map)
generate_comparison_plots(distance_metrics, organ_names_d, 0, 500, titlemap_distance, ylabelmap_distance, basedir, basedir2, paperplots_dir, color_map)

########## Centerline Dice plots ##########
organ_names_cl = ['Esophagus', 'Aorta']
cldice_metrics = ['cldice']
titlemap_cldice = {'cldice': 'Centerline Dice - Validation 25 Epochs'}
ylabelmap_cldice = {'cldice': 'clDice'}
generate_plots(cldice_metrics, organ_names_cl, 0, 1, titlemap_cldice, ylabelmap_cldice, basedir, paperplots_dir, color_map)
generate_comparison_plots(cldice_metrics, organ_names_cl, 0, 1, titlemap_cldice, ylabelmap_cldice, basedir, basedir2, paperplots_dir, color_map)
