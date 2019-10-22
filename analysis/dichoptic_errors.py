import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy import stats

# Directories
main_dir = "C:/Users/angie/Git Root/StereoTraining/GameObservers/"
sub_dir = "/DartBoard/"
results_dir = "C:/Users/angie/Git Root/StereoTraining/figs/errors/"

# Load and read data from csv files
os.chdir('C:/Users/angie/Git Root/StereoTraining/data')
allData = pd.read_csv('subjectData.csv')
dataFit = pd.read_csv('stanFit.csv')

# Set global lists
subjects = allData.subject.unique()
difficulties = allData.difficulty.unique()

# Plot parameters
regressColors = ['#CC0000', '#66CC00', '#0000CC'] # Plot colors
regressXvals = 6.1
regressYvals = [1.4, 1.3, 1.2] # Text y position

dataFrame = allData.drop(columns=["SA[seconds] dart location", "distance[m]", "gapAngle[degrees]", "head Hor Mean",
                                  "head Ver Mean", "head Ver SD", "head Hor SD", "head Z Mean", "head Z SD"])
dataFrame = dataFrame.rename(columns={'dichoptic errors' : 'dichoptic_errors'})

for row in dataFrame.index:
    if dataFrame.dichoptic_errors[row] >= 3:
        dataFrame.dichoptic_errors[row] = 3

def getErrors():
    #variables of interest
    error_subject = []
    error_date = []
    session_error_ratio = []

    for sub in subjects:
        subData = dataFrame.loc[dataFrame.subject == sub]
        dates = subData.date.unique()

        for date in dates:
            total_errs = np.sum(subData.dichoptic_errors[subData.date == date])

            if total_errs == 0:
                total_errors = 0
            else:
                total_errors = total_errs / len(subData.dichoptic_errors[subData.date == date])

            error_subject.append(sub)
            error_date.append(date)
            session_error_ratio.append(total_errors)

    return error_subject, error_date, session_error_ratio

subject_list, date_list, session_error_ratio_list = getErrors()
d = {'subject': subject_list, 'date': date_list, 'session_errors': session_error_ratio_list}
df = pd.DataFrame(d)

def getRatio():
    ratio_sub = []
    ratio = []
    total_errors = []

    for sub in subjects:
        err_ratio = (np.sum(df.session_errors[df.subject == sub][0:8])/8) / (np.sum(df.session_errors[df.subject == sub][-8:])/8)
        total_err = np.sum(df.session_errors[df.subject == sub][0:]) / len(df.session_errors[df.subject == sub])

        ratio_sub.append(sub)
        ratio.append(err_ratio)
        total_errors.append(total_err)

    return ratio_sub, ratio, total_errors

sub_list, ratio_list, total_error_list = getRatio()
d2 = {'subject': sub_list, 'errRatio': ratio_list, 'total_errors': total_error_list}
df2 = pd.DataFrame(d2)

df3 = pd.merge(dataFit, df2, on='subject')

def getPlotParams():
    slope = []
    intercept = []
    r_val = []
    p_val = []
    std_err = []
    line = []

    for diff in difficulties:

        # Generate linear fit
        # Regression variables
        regression_x = df3.errRatio[df3.difficulty == diff]
        regression_y = df3.ppr[df3.difficulty == diff]

        slope1, intercept1, r_val1, p_val1, std_err1 = stats.linregress(regression_x, regression_y)
        line1 = slope1 * regression_x + intercept1

        slope.append(slope1)
        intercept.append(intercept1)
        r_val.append(r_val1)
        p_val.append(p_val1)
        std_err.append(std_err1)
        line.append(line1)

    return slope, intercept, r_val, p_val, std_err, line

slope, intercept, r_val, p_val, std_err, line = getPlotParams()

# Is there a correlation between the amount of in-game learning and the ratio of errors?
font = {'weight': 'bold', 'size': 18}
matplotlib.rc('font', **font)
sns.set('poster')
sns.set_style('whitegrid')

sns.scatterplot(x='errRatio', y='ppr', hue='difficulty', data=df3, alpha=0.65, palette=[regressColors[0], regressColors[1], regressColors[2]])
plt.xlabel('Error ratio')
plt.ylabel('Learning (ppr)')

L = plt.legend(bbox_to_anchor=(0.06, 1.01), loc=2, borderaxespad=0., prop={'size':10}, frameon= False)
L.get_texts()[0].set_text('')
L.get_texts()[1].set_text('Level 1: All cues')
L.get_texts()[2].set_text('Level 2: Motion parallax & disparity')
L.get_texts()[3].set_text('Level 3: Disparity only')

for diff in difficulties:
    for j in range(0,3):
        plt.plot(df3.errRatio[df3.difficulty == diff], line[j], '-', color= regressColors[j])
        plt.text(regressXvals, regressYvals[j], 'r = ' + str('%.2f' % r_val[j]) + ',   ' + 'p = ' + str('%.2f' % p_val[j]), fontsize=8,
                 color=regressColors[j])

#plt.show()

name = "Learning to error ratio.png"
plt.savefig(fname=results_dir + name, bbox_inches='tight', format='png', dpi=300)

# for sub in subjects:
#     data = []
#     name = sub + "_dichoptic_errors.png"
#     data = df.loc[df.subject == sub]
#
#     font = {'weight': 'bold', 'size': 18}
#     matplotlib.rc('font', **font)
#     sns.set('poster', palette='colorblind')
#     sns.set_style('whitegrid')
#
#     ax = sns.scatterplot(x='date', y='errors', data=data, alpha=.85)
#
#     # ax = sns.scatterplot(x='date', y='errors', hue='difficulty', data=data,
#     #                      palette=sns.color_palette('colorblind', n_colors=3), alpha=.85)
#     ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
#
#     ax.set(xticklabels=np.arange(1, len(data.date.unique()), step=10),
#            xlabel='Session number', ylabel='dichoptic errors')
#     plt.xticks(np.arange(0, len(data.date.unique()), step=10))
#
#     plt.title(sub)
#
#     plt.savefig(fname=results_dir + name, bbox_inches='tight', format='png', dpi=300)
#     plt.clf()
#     plt.cla()