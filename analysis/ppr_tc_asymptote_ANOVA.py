# 2x3 way between subjects ANOVA on TC, Asymptote, and PPR

import pandas as pd
import os
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import scipy

main_dir = "C:/Users/angie/Git Root/StereoTraining/GameObservers/"
sub_dir = "/DartBoard/"
results_dir = "C:/Users/angie/Git Root/StereoTraining/figs/"
Control = {'ah', 'aj', 'dd', 'dl', 'ez', 'it', 'll', 'sh', 'sm', 'sr'} #10 control
Anomalous = {'bb', 'by', 'co', 'et', 'jz', 'kp', 'ky', 'mb', 'mg', 'tp'}  #10 experimental
obs_set = {'ah', 'aj', 'dd', 'dl', 'ez', 'it', 'll', 'sh', 'sm', 'sr', 'bb', 'by', 'co', 'et', 'jz', 'kp', 'ky', 'mb', 'mg', 'tp'}

# Load variables
os.chdir('C:/Users/angie/Git Root/StereoTraining/data')
allData = pd.read_csv('boxPlotData.csv')

subjects = allData.subject.unique()
difficulties = allData.difficulty.unique()

#For some reason, this gives me wrong dof..
ols_lm = smf.ols('tc ~ difficulty * group', data=allData)
fit = ols_lm.fit()
table = sm.stats.anova_lm(fit, typ=2)
#print(table)

# Analysis for Pre to Post ratio (PPR)
def computeP(fValue, df_diff, df_group):
    fDistribution = scipy.stats.f
    p = 1 - fDistribution.cdf(fValue, df_diff, df_group)
    return p

def omnibusTest(result):
    model = result[:-1]
    residual = result.iloc[-1]
    dof = np.sum(model['df'])
    meansquare_explained = np.sum(model['sum_sq'])/dof
    meansquare_unexplained = residual['sum_sq']/residual['df']
    F = meansquare_explained/meansquare_unexplained
    p = computeP(F, dof, residual['df'])
    return (F,p)

F,p = omnibusTest(table)
#print((F,p))

def geoMean(data_set):
    data_length = len(data_set)
    geometricMean = pow(np.prod(data_set), 1/data_length)
    return geometricMean

anovaFrame = allData.copy()
#print(anovaFrame.sample(10))

##Hard Way
#Calculate 2-way anova
# DEGREES OF FREEDOM
N = len(allData['tc'])
df_diff = len(allData['difficulty'].unique()) - 1
df_group = len(allData['group'].unique()) - 1
df_diffxgroup = df_diff*df_group
df_w = N - (len(allData['difficulty'].unique())*len(allData['group'].unique()))

# SUMS OF SQUARES
dataMean = anovaFrame['tc'].mean()
groupMeans = anovaFrame.groupby('group').tc.mean()
diffMeans = anovaFrame.groupby('difficulty').tc.mean()
groupXdiffMeans = anovaFrame.groupby(['group', 'difficulty']).tc.mean()

#Calculate geometric mean
geoAnolMean = geoMean(allData.tc[allData['group'] == 1])
geoNorMean = geoMean(allData.tc[allData['group'] == 0])

#Calculate SEM
semAnol = scipy.stats.sem(allData.tc[allData['group'] == 1])
semNorm = scipy.stats.sem(allData.tc[allData['group'] == 0])

print('Anomalous geometric mean: ', geoAnolMean, 'SEM: ', semAnol)
print('Normal geometric mean: ', geoNorMean, 'SEM: ', semNorm)

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
anovaFrame['residual'] = anovaFrame.tc - anovaFrame.meanEffect - anovaFrame.groupMainEffect - anovaFrame.difficultyMainEffect - anovaFrame.interactionEffect
#print(anovaFrame.sample(10))

# SUMS OF SQUARES
def SS(x):
    return np.sum(np.square(x))

sumofSquares = {}
keys = ['total', 'mean', 'group', 'difficulty', 'interaction', 'residual']
columns = [anovaFrame.tc, anovaFrame.meanEffect, anovaFrame.groupMainEffect, anovaFrame.difficultyMainEffect, anovaFrame.interactionEffect, anovaFrame['residual']]
for key, column, in zip(keys, columns):
    sumofSquares[key] = SS(column)
print('Sums of Squares: ', sumofSquares)

# Calculate degrees of freedom for sums of squares and  store it into a dictionary
dof = {}
vals = [N, 1,
        len(group)-1,
        len(difficulties)-1,
        (len(difficulties)-1)*(len(group)-1),
        N-len(group)*len(difficulties)]

for key, val in zip(keys, vals):
    dof[key] = val
print('Degrees of Freedom: ', dof)

# Using the dictionaries "sumsofSquares" and "dof", compute the mean square values dor all of the keyed quantities
meanSquare = {}
for key in keys:
    meanSquare[key] = sumofSquares[key]/dof[key]
print('Mean Square: ')
print(meanSquare)

# Compute the F statistic for each main effect and interaction
F = {}
for key in ['group', 'difficulty', 'interaction']:
    F[key] = meanSquare[key]/meanSquare['residual']
print('F value: ', F)

#Compute p value using scipy.stats.f.cdf
for effect in F.keys():
    print(effect)
    print("\t"+str(computeP(F[effect], dof[effect], dof['residual'])))

allData
