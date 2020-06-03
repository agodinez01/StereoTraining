## AG 12/21/19

# TODO:

import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import scipy.stats

#Run this if you have to create a new dataframe.
main_dir = "C:/Users/angie/Git Root/StereoTraining/GameObservers/"
results_dir = "C:/Users/angie/Git Root/StereoTraining/figs/hitRate/"

# Load variables
os.chdir('C:/Users/angie/Git Root/StereoTraining/data')
stereoData = pd.read_csv('stereoTests.csv')

# Define lists
subjects = stereoData.subject.unique()

stereoPreAnol = stereoData.value[(stereoData.group == 'anomalous') & (stereoData.order == 'pre')].tolist()
stereoPostAnol = stereoData.value[(stereoData.group == 'anomalous') & (stereoData.order == 'post')].tolist()

sns.distplot(stereoPreAnol, color='blue')
sns.distplot(stereoPostAnol, color='red')
#plt.hist(stereoPostAnol, color='red')
#plt.hist(stereoPreAnol, color='blue')
plt.show()
plt.clf()

preAnolSA = np.mean(np.array(stereoPreAnol))
preAnolSD = np.std(np.array(stereoPreAnol))
print('Pre anomalous SA: ')
print(preAnolSA)
print(preAnolSD)
postAnolSA = np.mean(np.array(stereoPostAnol))
postAnolSD = np.std(np.array(stereoPostAnol))
print('Post anomalous SA: ')
print(postAnolSA)
print(postAnolSD)

stat, p = scipy.stats.wilcoxon(stereoPreAnol, stereoPostAnol)

stereoPreNor = stereoData.value[(stereoData.group == 'control') & (stereoData.order == 'pre')].tolist()
stereoPostNor = stereoData.value[(stereoData.group == 'control') & (stereoData.order == 'post')].tolist()

sns.distplot(stereoPreNor, color='blue')
sns.distplot(stereoPostNor, color='red')
plt.show()

preNorSA = np.mean(np.array(stereoPreNor))
preNorSD = np.std(np.array(stereoPreNor))
print('Pre normal SA: ')
print(preNorSA)
print(preNorSD)
postNorSA = np.mean(np.array(stereoPostNor))
postNorSD = np.std(np.array(stereoPreNor))
print('Post normal SA: ')
print(postAnolSA)
print(postNorSD)

statN, pN = scipy.stats.wilcoxon(stereoPreNor, stereoPostNor)

stereoData