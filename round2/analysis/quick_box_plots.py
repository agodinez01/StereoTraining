import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pingouin as pg

withImage_anova = 'yes' # takes 'yes' or 'no' depending on whether code that runs the figure snd ANOVA should be executed.
run_type = 'CSF_acuity'     # takes 'CSF_acuity' or 'AULCSF' depending on the variable of interest. 'area' for area under the CSF curve and 'acuity' for CSF acuity

main_dir = "C:/Users/angie/Git Root/StereoTraining/round2/data/"
results_dir = "C:/Users/angie/Git Root/StereoTraining/round2/figs/"

# Load variables
os.chdir('C:/Users/angie/Git Root/StereoTraining/round2')
data = pd.read_csv('ExponentialAdjustment.csv')

type = ['asympThreshold', 'asympPPR', 'time', 'PPR']

for i in type:

    # Plot params
    font = {'weight': 'bold', 'size': 18}
    matplotlib.rc('font', **font)
    sns.set('poster', palette='colorblind')
    sns.set_style('whitegrid')

    ax = sns.boxplot(x='condition', y=i, data=data, hue='group')
    #ax.set_yscale('log')
    #ax.set_ylim(50, 850)
    #ax.set_yticks([100, 300, 500, 800])
    #ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    name = i + '.png'

    plt.savefig(fname=results_dir + name, bbox_inches='tight', format='png', dpi=300)

    plt.clf()

