import numpy as np
from scipy.stats import ttest_ind, levene, wilcoxon


def ttest(data1, data2):
    '''
    Perform Welch t test
    '''
    # Calculate the significance
    value, pvalues = ttest_ind(data1, data2, equal_var=False)

    for pvalue in pvalues:
        if pvalue > 0.05:
            print('Samples are likely drawn from the same distributions (fail to reject H0)')
        else:
            print('Samples are likely drawn from different distributions (reject H0)')


def leveneTest(data_list):
    '''
    Check for variance difference between 2 or more data points
    '''
    if len(data_list) == 2:
        res = levene(data_list[0], data_list[1])
        print(res.statistic)
    elif len(data_list) == 3:
        res = levene(data_list[0], data_list[1], data_list[2])
        print(res.statistic)
    elif len(data_list) == 4:
        res = levene(data_list[0], data_list[1], data_list[2], data_list[3])
        print(res.statistic)
    elif len(data_list) == 5:
        res = levene(data_list[0], data_list[1], data_list[2], data_list[3], data_list[4])
        print(res.statistic)


def wilcoxTest(data_1, data_2):
    print(wilcoxon(data_1, data_2, alternative='greater'))


# Load data
values1 = np.load('FINALBASELINE\95HD_val.npy')
values2 = np.load('ERG3C2B\95HD_val.npy')

# Load best epoch
f = open("FINALBASELINE\\best_epoch.txt", "r")
best_epoch_value_b = int(f.readline())
values1 = values1[best_epoch_value_b]

f = open("ERG3C2B\\best_epoch.txt", "r")
best_epoch_value_pp = int(f.readline())
values2 = values2[best_epoch_value_pp]

ttest(values1, values2)

## Get specific class
# base = values1.reshape(-1, values1.shape[2])
esophogus = values1[0:values1.shape[0], 0]
heart = values1[0:values1.shape[0], 1]
trachea = values1[0:values1.shape[0], 2]
aorta = values1[0:values1.shape[0], 3]

# pp = values2.reshape(-1, values2.shape[2])
esophogusp = values2[0:values2.shape[0], 0]
heartp = values2[0:values2.shape[0], 1]
tracheap = values2[0:values2.shape[0], 2]
aortap = values2[0:values2.shape[0], 3]

t1 = esophogus.tolist()
t2 = esophogusp.tolist()

leveneTest([t1 , t2])
wilcoxTest(t1, t2)
