import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import scipy

results_dir = "C:/Users/angie/Git Root/StereoTraining/round2/figs/"
Control = {'ah', 'aj', 'dd', 'dl', 'ez', 'it', 'll', 'sh', 'sm', 'sr'} #10 control
Anomalous = {'bb', 'by', 'co', 'et', 'jz', 'kp', 'ky', 'mb', 'mg', 'tp'}  #10 experimental
obs_set = {'ah', 'aj', 'dd', 'dl', 'ez', 'it', 'll', 'sh', 'sm', 'sr', 'bb', 'by', 'co', 'et', 'jz', 'kp', 'ky', 'mb', 'mg', 'tp'}

# Load variables
os.chdir('C:/Users/angie/Git Root/StereoTraining/round2/data')
allData = pd.read_csv('subjectData_all.csv')

subjects = allData.subject.unique()

highest = allData.stereoacuity.max()

for index, item in enumerate(allData.stereoacuity):
    if item == 0:
        allData.stereoacuity[index] = highest
        allData.saLog[index] = np.log10(highest)

allData

def add_id():
    idVals = []
    conditionVals = []

    for sub in subjects:
        data = allData.loc[allData.subject == sub]

        for item in data.subject:

            if sub == 'ah':
                id = 'N1'
                conditionVal = 'normal'
            elif sub == 'aj':
                id = 'N2'
                conditionVal = 'normal'
            elif sub == 'dd':
                id = 'N3'
                conditionVal = 'normal'
            elif sub == 'dl':
                id = 'N4'
                conditionVal = 'normal'
            elif sub == 'ez':
                id = 'N5'
                conditionVal = 'normal'
            elif sub == 'it':
                id = 'N6'
                conditionVal = 'normal'
            elif sub == 'll':
                id = 'N7'
                conditionVal = 'normal'
            elif sub == 'sh':
                id = 'N8'
                conditionVal = 'normal'
            elif  sub == 'sm':
                id = 'N9'
                conditionVal = 'normal'
            elif sub == 'sr':
                id = 'N10'
                conditionVal = 'normal'
            elif sub == 'by':
                id = 'AA1'
                conditionVal = 'anisometropia'
            elif sub == 'co':
                id = 'AA2'
                conditionVal = 'anisometropia'
            elif sub == 'ky':
                id = 'AA3'
                conditionVal = 'anisometropia'
            elif sub == 'mb':
                id = 'AA4'
                conditionVal = 'anisometropia'
            elif sub == 'mg':
                id = 'AMS1'
                conditionVal = 'strabismus'
            elif sub == 'bb':
                id = 'AS1'
                conditionVal = 'strabismus'
            elif sub == 'et':
                id = 'AS2'
                conditionVal = 'strabismus'
            elif sub == 'kp':
                id = 'AS3'
                conditionVal = 'strabismus'
            elif sub == 'tp':
                id = 'AS4'
                conditionVal = 'strabismus'
            elif sub == 'jz':
                id = 'ASW1'
                conditionVal = 'stereo-weak'



            idVals.append(id)
            conditionVals.append(conditionVal)

    return idVals, conditionVals

idVals, conditionVals = add_id()

allData['id'] = idVals
allData['condition'] = conditionVals

allData = allData[['subject', 'id', 'group', 'condition', 'date', 'difficulty', 'hit', 'stereoacuity', 'saLog', 'dichoptic errors', 'SA[seconds] dart location' , 'distance[m]',
                   'gapAngle[degrees]', 'head Hor Mean', 'head Hor SD', 'head Ver Mean', 'head Ver SD', 'head Z Mean', 'head Z SD']]


allData.to_csv(r'C:\Users\angie\Git Root\StereoTraining\round2\data\dartBoardRaw.csv', index=False)





