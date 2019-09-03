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
allData = pd.read_csv('subjectData.csv')

allData = allData.ix[(allData['hit']== True) & (allData['stereoacuity'] > 50)]

subjects = allData.subject.unique()
difficulties = allData.difficulty.unique()

#For some reason, this gives me wrong dof..
ols_lm = smf.ols('saLog ~ difficulty * group', data=allData)
fit = ols_lm.fit()
table = sm.stats.anova_lm(fit, typ=2)
print(table)

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
print((F,p))

anovaFrame = allData.copy()
print(anovaFrame.sample(10))

##Hard Way
#Calculate 2-way anova
# DEGREES OF FREEDOM
N = len(allData['saLog'])
df_diff = len(allData['difficulty'].unique()) - 1
df_group = len(allData['group'].unique()) - 1
df_diffxgroup = df_diff*df_group
df_w = N - (len(allData['difficulty'].unique())*len(allData['group'].unique()))

anovaFrame = allData.drop(columns=['SA[seconds] dart location', 'dichoptic errors', 'distance[m]', 'subject'])

# SUMS OF SQUARES
dataMean = anovaFrame['saLog'].mean()
groupMeans = anovaFrame.groupby('group').saLog.mean()
diffMeans = anovaFrame.groupby('difficulty').saLog.mean()
groupXdiffMeans = anovaFrame.groupby(['group', 'difficulty']).saLog.mean()

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
anovaFrame['residual'] = anovaFrame.saLog - anovaFrame.meanEffect - anovaFrame.groupMainEffect - anovaFrame.difficultyMainEffect - anovaFrame.interactionEffect
print(anovaFrame.sample(10))

# SUMS OF SQUARES
def SS(x):
    return np.sum(np.square(x))

sumofSquares = {}
keys = ['total', 'mean', 'group', 'difficulty', 'interaction', 'residual']
columns = [anovaFrame.saLog, anovaFrame.meanEffect, anovaFrame.groupMainEffect, anovaFrame.difficultyMainEffect, anovaFrame.interactionEffect, anovaFrame['residual']]
for key, column, in zip(keys, columns):
    sumofSquares[key] = SS(column)
print('Sums of Squares: ')
print(sumofSquares)

# Calculate degrees of freedom for sums of squares and  store it into a dictionary
dof = {}
vals = [N, 1,
        len(group)-1,
        len(difficulties)-1,
        (len(difficulties)-1)*(len(group)-1),
        N-len(group)*len(difficulties)]

for key, val in zip(keys, vals):
    dof[key] = val
print('Degrees of Freedom: ')
print(dof)

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
print('F value: ')
print(F)

#Compute p value using scipy.stats.f.cdf
for effect in F.keys():
    print(effect)
    print("\t"+str(computeP(F[effect], dof[effect], dof['residual'])))

allData
