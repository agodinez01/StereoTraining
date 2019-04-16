import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import scipy

main_dir = "C:/Users/angie/Git Root/StereoTraining/GameObservers/"
sub_dir = "/DartBoard/"
results_dir = "C:/Users/angie/Git Root/StereoTraining/violinPlot/figs/"
Control = {'ah', 'aj', 'dd', 'dl', 'ez', 'it', 'll', 'sh', 'sm', 'sr'} #10 control
Anomalous = {'bb', 'by', 'co', 'et', 'gn', 'gp', 'jz', 'kp', 'ky', 'mb', 'mg', 'ni', 'tp'}  #13 experimental
obs_set = {'ah', 'aj', 'dd', 'dl', 'ez', 'it', 'll', 'sh', 'sm', 'sr', 'bb', 'by', 'co', 'et', 'gn', 'gp', 'jz', 'kp', 'ky', 'mb', 'mg', 'ni', 'tp'}

def get_yvals(obs_set):
    subData = []
    for obs in obs_set:
        path = main_dir + obs + sub_dir
        for data in os.listdir(path):
            if not data.endswith('.csv'):
                continue

            df = pd.read_csv(path + data)
            dataf = df
            dataf['subject'] = obs
            dataf['date'] = data[11:]

            if obs_set == Anomalous:
                dataf['group'] = 'Anomalous'
            elif obs_set == Control:
                dataf['group'] = 'Control'

            subData.append(dataf)
    return subData

controlData = get_yvals(Control)
controlAll = pd.concat(controlData, sort= True)

anolData = get_yvals(Anomalous)
anolAll = pd.concat(anolData, sort= True)

frames = [controlAll, anolAll]
allData = pd.concat(frames)
allData.rename(columns={'SA[seconds] dartboard hit':'stereoacuity', 'Difficulty':'difficulty'}, inplace=True)

grouped = allData.groupby(['group', 'difficulty', 'hit'])
hitRate = grouped["hit"].count()

grouped2 = allData.groupby(['group', 'difficulty'])
hitTotal = grouped2['hit'].count()

anolHitRate = hitRate.loc['Anomalous', [1, 2, 3], True] / hitTotal['Anomalous']
controlHitRate = hitRate.loc['Control', [1, 2, 3], True] / hitTotal['Control']

grouped3 = allData.groupby(['subject', 'difficulty', 'hit'])
subjectHit = grouped3['hit'].count()

grouped4 = allData.groupby(['subject', 'difficulty'])
subjectHitTotal = grouped4['hit'].count()

grouped5 = allData.groupby(['subject', 'date', 'difficulty'])
saMean = grouped5['stereoacuity'].mean()
saMedian = grouped5['stereoacuity'].median()

subjects = allData.subject.unique()
difficulties = allData.difficulty.unique()

def getSa():
    saMed = []
    saSub = []
    saDate = []
    saDiff = []
    saSEM = []
    saLog = []

    for sub in subjects:
        subData = allData.loc[allData.subject==sub]
        subData = subData.loc[subData['stereoacuity']>50]
        dates = subData.date.unique()

        for date in dates:
            for difficulty in difficulties:
                subjectSaMedian = subData.stereoacuity[(subData.date==date) & (subData.difficulty==difficulty)].median()
                subjectSEM = subData.stereoacuity[(subData.date==date) & (subData.difficulty==difficulty)].sem()
                subjectLog = np.log10(subjectSaMedian)

                saSub.append(sub)
                saDate.append(date)
                saDiff.append(difficulty)

                saMed.append(subjectSaMedian)
                saSEM.append(subjectSEM)
                saLog.append(subjectLog)

    return  saSub, saDate, saDiff, saMed, saSEM, saLog

subject_list, date_list, difficulty_list, saMed_list, saSEM_list, saLog_list = getSa()
d = {'subject':subject_list, 'date':date_list, 'difficulty':difficulty_list, 'saMedian':saMed_list, 'saSEM':saSEM_list, 'saLog':saLog_list}
df = pd.DataFrame(d)

def getNormalizer():
    normTerm = []
    normSub = []
    normDiff = []

    for sub in subjects:
        for diff in difficulties:
            nTerm = df.saLog[(df.subject==sub) & (df.difficulty==diff)][-10:].mean()
            normTerm.append(nTerm)

            normSub.append(sub)
            normDiff.append(diff)
    return normTerm, normSub, normDiff

normTerm, normSub, normDiff = getNormalizer()

normTable = {'subject':normSub, 'difficulty':normDiff, 'normalizer':normTerm}
normDf = pd.DataFrame(normTable)

def addNormalizer():
    normData = []
    subject = []
    difficulty = []

    for sub in subjects:
        for diff in difficulties:
            saLogArray = df.saLog[(df.subject==sub) & (df.difficulty==diff)]
            normalizer = normDf.normalizer[(normDf.subject==sub) & (normDf.difficulty==diff)]
            normalized = np.divide(saLogArray, normalizer)
            normData.append(normalized)

            subject1 = [sub] * len(normalized)
            subject1 = pd.Series(subject1)
            subject.append(subject1)

            difficulty1 = [diff] * len(normalized)
            difficulty1 = pd.Series(difficulty1)
            difficulty.append(difficulty1)

            df['normalized']

    return normData, subject, difficulty
normData, normDataSub, normDataDiff  = addNormalizer()

normDataTable = {'subject':normDataSub, 'difficulty':normDataDiff, 'normalized':normData}
normData = pd.DataFrame(normDataTable)

def func(x, a, b, c):
    return a * np.exp(-b * x) +c

for sub in subjects:
    plt.figure(1)
    plt.subplot(2, 1, 1)

    ax = sns.scatterplot(x='date', y='saMedian', data=df.loc[df['subject']==sub], hue='difficulty', palette=sns.color_palette('colorblind', n_colors=3))
    plt.errorbar(x='date', y='saMedian', data=df.loc[df['subject']==sub], yerr='saSEM', fmt='none', hue='difficulty')

    ax.set(xticklabels=np.arange(1,len(df.loc[df['subject']==sub].date.unique()), step=10), yticklabels=np.arange(50, 850, step=100), ylabel='Stereoacuity (arc secs)', xlabel=None)
    plt.yticks(np.arange(50, 850, step=100))
    plt.xticks(np.arange(0,len(df.loc[df['subject']==sub].date.unique()), step=10))
    plt.title("Learning curve: " + sub.upper())

    plt.subplot(2, 1, 2)
    ax1 = sns.scatterplot(x='date', y='saLog', data=df.loc[df['subject']==sub], hue='difficulty', palette=sns.color_palette('colorblind', n_colors=3), legend=False)
    ax1.set(xticklabels=np.arange(1,len(df.loc[df['subject']==sub].date.unique()), step=10), xlabel='Session number')
    plt.yticks(np.arange(2, 4, step=1))
    plt.xticks(np.arange(0, len(df.loc[df['subject'] == sub].date.unique()), step=10))

    plt.show()

    name = sub+"_learningCurve.jpeg"
    plt.savefig(fname=results_dir + name, bbox_inches='tight', format='png', dpi=None)
    plt.clf()

def getHit(var1, var2):
    result = []
    for sub in obs_set:
        hitRate = var1.loc[sub, [1, 2, 3], True] / var2[sub]
        result.append(hitRate)
    return result

hitRateBySub = getHit(subjectHit, subjectHitTotal)

dataSavePath = main_dir + 'processed_data.pkl'
allData.to_pickle(dataSavePath)

## SA performance
plt.figure()
# sns.set_context("poster")
ax = sns.boxplot(x='difficulty', y='stereoacuity', data=allData, hue='group', palette='colorblind', fliersize=0)
ax.set(ylabel='Stereoacuity (arc secs)', xticklabels=['Natural', 'Advanced', 'Expert'])
ax.set_ylim(bottom=None, top=1500)
plt.legend(loc=1)

plt.show()

name = "Effect of group and difficulty level on stereoacuity.jpeg"
plt.savefig(fname=results_dir + name, bbox_inches='tight', format='pdf', dpi=1000)

allData = allData.ix[(allData['hit']== True) & (allData['stereoacuity'] > 50)]

#For some reason, this gives me wrong dof..
ols_lm = smf.ols('stereoacuity ~ difficulty * group', data=allData)
fit = ols_lm.fit()
anov_table = sm.stats.anova_lm(fit, typ=2)
anov_table

def computeP(fValue, df_diff, df_group):
    fDistribution = scipy.stats.f
    p = 1 - fDistribution.cdf(fValue, df_diff, df_group)
    return p

##Hard Way
#Calculate 2-way anova
# DEGREES OF FREEDOM
N = len(allData['stereoacuity'])
df_diff = len(allData['difficulty'].unique()) - 1
df_group = len(allData['group'].unique()) - 1
df_diffxgroup = df_diff*df_group
df_w = N - (len(allData['Difficulty'].unique())*len(allData['group'].unique()))

anovaFrame = allData.drop(columns=['SA[seconds] dart location', 'dichoptic errors', 'distance[m]', 'subject'])

# SUMS OF SQUARES
dataMean = anovaFrame['stereoacuity'].mean()
groupMeans = anovaFrame.groupby('group').stereoacuity.mean()
diffMeans = anovaFrame.groupby('difficulty').stereoacuity.mean()
groupXdiffMeans = anovaFrame.groupby(['group', 'difficulty']).stereoacuity.mean()

anovaFrame["meanEffect"] = dataMean
group = allData.group.unique()

for g in group:
    selector = (anovaFrame.group == g)
    anovaFrame.loc[selector, "groupMainEffect"] = groupMeans.loc[g]- dataMean

for diff in difficulties:
    selector = (anovaFrame.Difficulty == diff)
    anovaFrame.loc[selector, "difficultyMainEffect"] = diffMeans.loc[diff] - dataMean

for g in group:
    for diff in difficulties:
        selector = (anovaFrame.Difficulty == diff) & (anovaFrame.group == g)
        anovaFrame.loc[selector, "interactionEffect"] = groupXdiffMeans.loc[g, diff] - anovaFrame.groupMainEffect[selector] - anovaFrame.difficultyMainEffect[selector] - dataMean

# ERROR/RESIDUAL
anovaFrame['residual'] = anovaFrame.stereoacuity - anovaFrame.meanEffect - anovaFrame.groupMainEffect - anovaFrame.difficultyMainEffect - anovaFrame.interactionEffect
anovaFrame.sample(10)

# SUMS OF SQUARES
def SS(x):
    return np.sum(np.square(x))

sumofSquares = {}
keys = ['total', 'mean', 'group', 'difficulty', 'interaction', 'residual']
columns = [anovaFrame.stereoacuity, anovaFrame.meanEffect, anovaFrame.groupMainEffect, anovaFrame.difficultyMainEffect, anovaFrame.interactionEffect, anovaFrame.residual]
for key, column, in zip(keys, columns):
    sumofSquares[key] = SS(column)
sumofSquares

# Calculate degrees of freedom for sums of squares and  store it into a dictionary
dof = {}
vals = [N, 1, len(difficulties)-1, len(group)-1, (len(difficulties)*len(group)-1), N-len(group)*len(difficulties)]
for key, val in zip(keys, vals):
    dof[key] = val
dof

# Using the dictionaries "sumsofSquares" and "dof", compute the mean square values dor all of the keyed quantities
meanSquare = {}
for key in keys:
    meanSquare[key] = sumofSquares[key]/dof[key]
meanSquare

# Compute the F statistic for each main effect and interaction
F = {}
for key in ['group', 'difficulty', 'interaction']:
    F[key] = meanSquare[key]/meanSquare['residual']
F

#Compute p value using scipy.stats.f.cdf
for effect in F.keys():
    print(effect)
    print("\t"+str(computeP(F[effect], dof[effect], dof['residual'])))

anov_table

#    x = np.arange(0, len(df.loc[df['subject']==sub].date.unique()))
# for diff in difficulties:
#     y = df.loc[(df['subject']==sub) & (df['difficulty']==diff)]['saMedian']
#     np.polyfit(np.log(x), y, 1)
#     a = len(x)
#     b = len(df.loc[(df['subject']==sub) & (df['difficulty']==diff)])


# np.polyfit(np.log(x), df.loc[df['subject']==sub]['saMedian'], 1)


# popt, pcov = curve_fit(func, df.loc[df['subject']==sub]['date'], df.loc[df['subject']==sub]['saMedian'])
# plt.plot(df.loc[df['subject']==sub]['date'], func(df.loc[df['subject']==sub]['date'], *popt), 'r-', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
# plt.plot(df.loc[df['subject']==sub]['date'], df.loc[df['subject']==sub]['saMedian'], 'r-', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
