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
quiverData = pd.read_csv('forQuiver.csv')

subjects = quiverData.Patient.unique()
nums = range(0,4)

def getDf():
    subVals = []
    conditionVals = []
    angleVals = []
    typeVals = []
    resultVals = []
    vals = []
    levelVals = []
    orderVals = []

    for sub in subjects:

        for i in nums:
            subC = sub
            #dfbb = df[df['A'] == 8].index.values.astype(int)[0]
            conditionC = quiverData.condition[quiverData.Patient == sub].tolist()[0]
            #conditionC = quiverData.condition.loc[quiverData.Patient == sub]
            angleC = quiverData.Angle.loc[quiverData.Patient == sub].tolist()[0]
            typeC = quiverData.Type.loc[quiverData.Patient == sub].tolist()[0]
            resultC = quiverData.Result.loc[quiverData.Patient == sub].tolist()[0]

            if i == 0:
                valC = quiverData.DepthPre[quiverData.Patient == sub].tolist()[0]
                levelC = 'level_1'
                orderC = 'pre'

            elif i == 1:
                valC = quiverData.DepthPost[quiverData.Patient == sub].tolist()[0]
                levelC = 'level_1'
                orderC = 'post'

            elif i == 2:
                valC = quiverData.StereoPre[quiverData.Patient == sub].tolist()[0]
                levelC = 'level_3'
                orderC = 'pre'

            elif i == 3:
                valC = quiverData.StereoPost[quiverData.Patient == sub].tolist()[0]
                levelC = 'level_3'
                orderC = 'post'

            subVals.append(subC)
            conditionVals.append(conditionC)
            angleVals.append(angleC)
            typeVals.append(typeC)
            resultVals.append(resultC)
            vals.append(valC)
            levelVals.append(levelC)
            orderVals.append(orderC)

    return subVals, conditionVals, angleVals, typeVals, resultVals, vals, levelVals, orderVals

subVals, conditionVals, angleVals, typeVals, resultVals, vals, levelVals, orderVals = getDf()
df = {'subject': subVals, 'condition': conditionVals, 'vals': vals, 'level': levelVals, 'order': orderVals, 'angle': angleVals, 'type': typeVals, 'result': resultVals}
dataF = pd.DataFrame(df)

plotColors = ['#595858', '#E91515', '#0923EF', '#0CB51F']  # Plot colors grey, red, blue and green
condition = dataF.condition.unique()

def GetQuiverVals(dataF, cond):
    color = []
    level1_X = []
    level3_Y = []
    u1 = []
    v1 = []

    df = dataF.loc[dataF.condition == cond]
    level1_X = df.vals[(df.order == 'pre') & (df.level == 'level_1')]
    level3_Y = df.vals[(df.order == 'pre') & (df.level == 'level_3')]

    u1 = np.subtract(df.vals[(df.order == 'post') & (df.level == 'level_1')],
                         df.vals[(df.order == 'pre') & (df.level == 'level_1')])
    v1 = np.subtract(df.vals[(df.order == 'post') & (df.level == 'level_3')],
                         df.vals[(df.order == 'pre') & (df.level == 'level_3')])

    if cond == 'binocular':
        color = plotColors[0]
    elif cond == 'strabismus':
        color = plotColors[1]
    elif cond == 'anisometropia':
        color = plotColors[2]
    elif cond == 'stereo-weak':
        color = plotColors[3]

    return level1_X, level3_Y, u1, v1, color

X1B, Y1B, u1B, v1B, colorB = GetQuiverVals(dataF, 'binocular')
X1S, Y1S, u1S, v1S, colorS = GetQuiverVals(dataF, 'strabismus')
X1A, Y1A, u1A, v1A, colorA = GetQuiverVals(dataF, 'anisometropia')
X1SW, Y1SW, u1SW, v1SW, colorSW = GetQuiverVals(dataF, 'stereo-weak')

font = {'weight': 'bold', 'size': 20}
matplotlib.rc('font', **font)
sns.set('poster', palette='colorblind')
sns.set_style('whitegrid')

plt.plot([175, 500], [175, 500], 'k--')

#plt.quiver(X1B, Y1B, u1B, v1B, label='Binocular', color=colorB, alpha=0.75)
plt.quiver(X1S, Y1S, u1S, v1S, label='Strabismus', color=colorS, alpha=0.75)
plt.quiver(X1A, Y1A, u1A, v1A, label='Anisometropia', color=colorA, alpha=0.75)
plt.quiver(X1SW, Y1SW, u1SW, v1SW, label='Stereo-weak', color=colorSW, alpha=0.75)

plt.ylabel('Disparity only (arc secs)')
plt.xlabel('All cues (arc secs)')

L = plt.legend(bbox_to_anchor=(1.01, 1.01), loc=2, borderaxespad=0, prop={'size': 10}, frameon=False)

name = 'QuiverPlotAnomalousOnly.png'
plt.savefig(fname=results_dir + name, bbox_inches='tight', format='png', dpi=300)

# ax.XAxis.Scale = 'log';
# ax.YAxis.Scale = 'log';
