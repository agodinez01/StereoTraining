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

clinicalPre = stereoData.value[(stereoData.group == 'anomalous') & (stereoData.order == 'pre') &
                               ((stereoData.test == 'randot') | (stereoData.test == 'random3'))].tolist()
clinicalPost = stereoData.value[(stereoData.group == 'anomalous') & (stereoData.order == 'post') &
                                ((stereoData.test == 'randot') | (stereoData.test == 'random3'))].tolist()
stat, p = scipy.stats.wilcoxon(clinicalPre, clinicalPost)

computerPre = stereoData.value[(stereoData.group == 'anomalous') & (stereoData.order == 'pre') &
                               ((stereoData.test == 'pdt') | (stereoData.test == 'drsS') | (stereoData.test == 'drsM') | (stereoData.test == 'drsB'))].tolist()
computerPost = stereoData.value[(stereoData.group == 'anomalous') & (stereoData.order == 'post') &
                                ((stereoData.test == 'pdt') | (stereoData.test == 'drsS') | (stereoData.test == 'drsM') | (stereoData.test == 'drsB'))].tolist()
stat2, p2 = scipy.stats.wilcoxon(computerPre, computerPost)

stereoData