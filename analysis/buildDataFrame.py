# Builds all-subjects data frame for DartBoard game.

import pandas as pd
import os
import numpy as np

main_dir = "C:/Users/angie/Git Root/StereoTraining/GameObservers/"
sub_dir = "/DartBoard/"
Control = {'ah', 'aj', 'dd', 'dl', 'ez', 'it', 'll', 'sh', 'sm', 'sr'} #10 control
Anomalous = {'bb', 'by', 'co', 'et', 'jz', 'kp', 'ky', 'mb', 'mg', 'tp'}  #13 experimental
obs_set = {'ah', 'aj', 'dd', 'dl', 'ez', 'it', 'll', 'sh', 'sm', 'sr', 'bb', 'by', 'co', 'et', 'jz', 'kp', 'ky', 'mb', 'mg', 'tp'}

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
allData.drop(columns='dichoptic errors')
allData['saLog'] = np.log10(allData.stereoacuity)

allData.to_csv(r'C:\Users\angie\Git Root\StereoTraining\data\subjectData.csv', index=False)

