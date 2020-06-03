import pandas as pd
import os
import pingouin as pg
import matplotlib.pyplot as plt
import numpy as np

main_dir = "C:/Users/angie/Git Root/StereoTraining/data"
results_dir = "C:/Users/angie/Git Root/StereoTraining/figs/"

# Load variables
os.chdir('C:/Users/angie/Git Root/StereoTraining/data')
importData = pd.read_csv('d_prime_PPR.csv')

#What is the variance between groups?
normalSD = np.var(importData.PPR[importData['group']=='normal'])
anomalousSD = np.var(importData.PPR[importData['group']=='anomalous'])

aov = importData.anova(dv='PPR', between=['group', 'level', 'sa'], ss_type=3)
print(aov)

#Bonferroni correction
pvals = [0.00003, 0.40907, 0.00578, 0.98322, 0.00929, 0.78624, 0.88409]
reject, pvals_corr = pg.multicomp(pvals, method='fdr_bh')
print(reject, pvals_corr)

a = importData.PPR[importData.group == 'normal']
b = importData.PPR[importData.group == 'anomalous']

hist = b.hist()
plt.show()
