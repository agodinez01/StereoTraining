import pandas as pd
import os
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

allData = allData.ix[(allData['hit']== True) & (allData['stereoacuity'] > 50)]

#For some reason, this gives me wrong dof..
ols_lm = smf.ols('stereoacuity ~ difficulty * group', data=allData)
fit = ols_lm.fit()
anov_table = sm.stats.anova_lm(fit, typ=2)
print(anov_table)

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
df_w = N - (len(allData['difficulty'].unique())*len(allData['group'].unique()))

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
    selector = (anovaFrame.difficulty == diff)
    anovaFrame.loc[selector, "difficultyMainEffect"] = diffMeans.loc[diff] - dataMean

for g in group:
    for diff in difficulties:
        selector = (anovaFrame.difficulty == diff) & (anovaFrame.group == g)
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