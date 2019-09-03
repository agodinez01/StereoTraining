import pandas as pd
import os
import math
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

main_dir = "C:/Users/angie/Git Root/StereoTraining/GameObservers/"
sub_dir = "/DartBoard/"
results_dir = "C:/Users/angie/Git Root/StereoTraining/figs/"

# Load variables
os.chdir('C:/Users/angie/Git Root/StereoTraining/data')
transferData = pd.read_csv('transferData.csv')
newDataFrame = transferData.loc[transferData.order == 'pre']
postSA = transferData.stereoacuity[transferData.order == 'post'].tolist()

newDataFrame['postSA'] = postSA
newDataFrame = newDataFrame.drop(columns='order')
df = newDataFrame.rename(columns={'stereoacuity': 'preSA'})

font = {'weight': 'bold', 'size': 20}
matplotlib.rc('font', **font)
sns.set('poster', palette='colorblind')
sns.set_style('whitegrid')

#sns.scatterplot(x='postSA', y='preSA', hue='test', data=df)
plt.xlabel('post stereoacuity (arc secs)')
plt.ylabel('pre-training stereoacuity (arc secs)')

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='xx-small')
plt.plot([0, max(df['postSA'])], [0, max(df['preSA'])], 'k--')
sns.scatterplot(x='postSA', y='preSA', data=df.loc[df.test == 'randot'], hue='subject')

plt.show()

subjects = transferData.subject.unique()

    #
    # plt.savefig(fname=results_dir + title + '.pdf', bbox_inches='tight',
    #             format='pdf', dpi=300)
    # plt.show()