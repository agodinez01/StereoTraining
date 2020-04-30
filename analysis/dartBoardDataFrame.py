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

allData = allData.loc[(allData['hit']== True) & (allData['stereoacuity'] > 50)]

subjects = allData.subject.unique()
difficulties = allData.difficulty.unique()

def makePrePostFrame():
    subV = []
    conditionV = []
    groupV = []
    #dateV = []
    difficultyV = []
    saMedianPreV = []
    saMedianPostV = []
    saMedianV = []
    saLogMedianV = []
    #saVarV = []
    #saLogV = []
    #saLogVarV = []

    for sub in subjects:
        subData = allData.loc[allData.subject == sub]
        dates = subData.date.unique()

        for level in difficulties:

            if sub == 'by' or sub == 'co' or sub == 'ky' or sub == 'mb':
                conditionVal = 'anisometropia'
                groupVal = 'anomalous'
            elif sub == 'bb' or sub == 'et' or sub == 'kp' or sub == 'tp':
                conditionVal = 'strabismus'
                groupVal = 'anomalous'
            elif sub == 'ah' or sub == 'aj' or sub == 'dd' or sub == 'dl' or sub == 'ez' or sub == 'it' or sub == 'll' or sub == 'sh' or sub == 'sm' or sub == 'sr':
                conditionVal = 'binocular'
                groupVal = 'normal'
            elif sub == 'mg' or sub == 'jz':
                conditionVal = 'stereo-weak'
                groupVal = 'anomalous'

            saMedianPre = subData.stereoacuity[
                (subData.difficulty == level) & (subData.date == dates[0]) | (subData.date == dates[1]) | (subData.date == dates[2]) | (
                            subData.date == dates[3]) | (subData.date == dates[4]) | (subData.date == dates[5])].median()

            saMedianPost = subData.stereoacuity[
                (subData.difficulty == level) & (subData.date == dates[-6]) | (subData.date == dates[-5]) | (
                            subData.date == dates[-4]) | (subData.date == dates[-3]) | (subData.date == dates[-2]) | (
                            subData.date == dates[-1])].median()

            saMedianVal = subData.stereoacuity[subData.difficulty == level].median()

            saLogMedianVal = np.log10(saMedianVal)

            subV.append(sub)
            conditionV.append(conditionVal)
            groupV.append(groupVal)
            difficultyV.append(level)
            saMedianPreV.append(saMedianPre)
            saMedianPostV.append(saMedianPost)
            saMedianV.append(saMedianVal)
            saLogMedianV.append(saLogMedianVal)
            #saVarV
            #saLogV = []
            #saLogVarV = []

    return subV, conditionV, groupV, difficultyV, saMedianPreV, saMedianPostV, saMedianV, saLogMedianV

subject_list, condition_list, group_list, difficulty_list, saMedianPre_list, saMedianPost_list, saMedian_list, saLogMedian_list = makePrePostFrame()
d = {'subject': subject_list, 'condition': condition_list, 'group': group_list, 'difficulty': difficulty_list, 'saMedianPre': saMedianPre_list, 'saMedianPost': saMedianPost_list, 'saMedian': saMedian_list, 'saLogMedian': saLogMedian_list}
df = pd.DataFrame(d)

df.to_csv(r'C:\Users\angie\Git Root\StereoTraining\data\dartBoardMedians.csv', index=False)

#allData

