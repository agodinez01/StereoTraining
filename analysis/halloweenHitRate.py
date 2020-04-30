import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

#Run this if you have to create a new dataframe.
main_dir = "C:/Users/angie/Git Root/StereoTraining/GameObservers/"
results_dir = "C:/Users/angie/Git Root/StereoTraining/figs/halloweenDelta/"

# Load variables
os.chdir('C:/Users/angie/Git Root/StereoTraining/data')
HallData = pd.read_csv('HalloweenSubjectData.csv')
hitRateData = pd.read_csv('HallHitRate4.csv')

# Define lists
subjects = hitRateData.subject.unique()
difficulty = hitRateData.difficulty.unique()
sa = hitRateData.sa.unique()

def makedf():

    subVals = []
    conditionVals = []
    diffVals = []
    saVals = []
    probRatio = []
    groupVals = []

    for sub in subjects:
        subData = hitRateData.loc[hitRateData.subject == sub]

        for diff in difficulty:
            for item in sa:

                meanProbabilityPre = (subData.probabilityCorrect[(subData.difficulty == diff) & (subData.sa == item)][0:6]/6).mean()
                meanProbabilityPost = (subData.probabilityCorrect[(subData.difficulty == diff) & (subData.sa == item)][-6:]/6).mean()

                if meanProbabilityPre == 'nan' or meanProbabilityPost =='nan':
                    probabilityRatio = 'nan'
                else:
                    probabilityRatio = meanProbabilityPost/meanProbabilityPre

                if sub[0:2] == 'AA':
                    cond = 'anisometropia'
                    group = 'anomalous'
                elif (sub[0:2] == 'AS') & (sub != 'ASW1'):
                    cond = 'strabismus'
                    group = 'anomalous'
                elif sub[0] == 'N':
                    cond = 'binocular'
                    group = 'binocular'
                elif sub == 'AMS1' or sub == 'ASW1':
                    cond = 'stereo-weak'
                    group = 'anomalous'

                subVals.append(sub)
                conditionVals.append(cond)
                diffVals.append(diff)
                saVals.append(item)
                probRatio.append(probabilityRatio)
                groupVals.append(group)

    return subVals, groupVals, conditionVals, diffVals, saVals, probRatio

subject_list, group_list, condition_list, difficulty_list, sa_list, probRatio_list = makedf()
frame = {'subject': subject_list, 'group': group_list, 'condition': condition_list, 'difficulty': difficulty_list, 'sa': sa_list, 'probRatio': probRatio_list}
dataF = pd.DataFrame(frame)

# Plot parameters
plotColors = ['#595858', '#E91515', '#0923EF', '#0CB51F']  # Plot colors grey, red, blue and green
plotMarkers = ['', 'o', 's', 'D']
legendNames = ['', ': All cues', ': Motion parallax & disparity', ': Disparity only']

for sub in subjects:

    ax = plt.figure()

    for diff in difficulty:

        if dataF.condition[dataF.subject == sub].tolist()[0] == 'binocular':
            #markerFill =[]
            markerFill = ['', 'none', plotColors[0], 'none']
            sns.regplot(x='sa', y='probRatio', color=plotColors[0], data=dataF[(dataF.subject == sub) & (dataF.difficulty == diff)],
                        marker=plotMarkers[diff], ci=False, label=diff, logx=True, scatter_kws={'facecolor':markerFill[diff]}, line_kws={'linewidth':3})
            plt.text(900, 1.3, sub.upper(), fontsize=16, color=plotColors[0])

        elif dataF.condition[dataF.subject == sub].tolist()[0] == 'strabismus':
            markerFill = ['', 'none', plotColors[1], 'none']
            sns.regplot(x='sa', y='probRatio', color=plotColors[1], data=dataF[(dataF.subject == sub) & (dataF.difficulty == diff)],
                        marker=plotMarkers[diff], ci=False, label=diff, logx=True, scatter_kws={'facecolor':markerFill[diff]}, line_kws={'linewidth':3})
            plt.text(900, 1.3, sub.upper(), fontsize=16, color=plotColors[1])

        elif dataF.condition[dataF.subject == sub].tolist()[0] == 'anisometropia':
            markerFill = []
            markerFill = ['', 'none', plotColors[2], 'none']
            sns.regplot(x='sa', y='probRatio', color=plotColors[2], data=dataF[(dataF.subject == sub) & (dataF.difficulty == diff)],
                        marker=plotMarkers[diff], ci=False, label=diff, logx=True, scatter_kws={'facecolor':markerFill[diff]}, line_kws={'linewidth':3})
            plt.text(900, 1.3, sub.upper(), fontsize=16, color=plotColors[2])

        elif dataF.condition[dataF.subject == sub].tolist()[0] == 'stereo-weak':
            markerFill = []
            markerFill = ['', 'none', plotColors[3], 'none']
            sns.regplot(x='sa', y='probRatio', color=plotColors[3], data=dataF[(dataF.subject == sub) & (dataF.difficulty == diff)],
                        marker=plotMarkers[diff], ci=False, label=diff, logx=True, scatter_kws={'facecolor':markerFill[diff]}, line_kws={'linewidth':3})
            plt.text(900, 1.3, sub.upper(), fontsize=16, color=plotColors[3])

    #ax.legend()
    L = plt.legend(bbox_to_anchor=(1.01, 1.01), loc=2, borderaxespad=0, prop={'size': 10}, frameon=False)
    L.get_texts()[0].set_text('Level 1: All cues')
    L.get_texts()[1].set_text('Level 2: Motion parallax & disparity')
    L.get_texts()[2].set_text('Level 3: Disparity only')

    plt.xlabel('Game disparity (arc secs)')
    plt.xticks((400, 600, 800, 1000))
    plt.yticks((0.8, 1, 1.2, 1.4))
    plt.ylabel('Stereo Pre:Post Ratio')
    #plt.show()
    name = sub + "_halloweenRaw.png"
    #plt.savefig(fname= results_dir + name, bbox_inches= 'tight', format= 'png', dpi= 300)
    #plt.show()

def makeBoxPlotFrame():
    subjectC = []
    groupC = []
    diffC = []
    probRatioC = []

    for sub in subjects:
        for diff in difficulty:

            pRatio = dataF.probRatio[(dataF.subject == sub) & (dataF.difficulty == diff)].mean()
            g = dataF.group[(dataF.subject == sub) & (dataF.difficulty == diff)].tolist()[0]

            subjectC.append(sub)
            groupC.append(g)
            diffC.append(diff)
            probRatioC.append(pRatio)

    return subjectC, groupC, diffC, probRatioC

subVals, groupVals, diffVals, probVals = makeBoxPlotFrame()
df2 = {'subject':subVals, 'group':groupVals, 'difficulty':diffVals, 'probRatioMean':probVals}
dataF2 = pd.DataFrame(df2)

dataF2



# def makedf(order):
#
#     subVals = []
#     diffVals = []
#     saVals = []
#     orderVals = []
#     ratio = []
#
#     for sub in subjects:
#         subData = hitRateData.loc[hitRateData.subject == sub]
#
#         for diff in difficulty:
#             for item in sa:
#
#                 if order == 0:
#                     meanRP = subData.ratio[(subData.difficulty == diff) & (subData.sa == item)][0:4].mean()
#                     o = 'pre'
#
#                     subVals.append(sub)
#                     diffVals.append(diff)
#                     saVals.append(item)
#                     orderVals.append(o)
#                     ratio.append(meanRP)
#
#                 elif order == 1:
#                     meanRP = subData.ratio[(subData.difficulty == diff) & (subData.sa == item)][-4:].mean()
#                     o = 'post'
#
#                     subVals.append(sub)
#                     diffVals.append(diff)
#                     saVals.append(item)
#                     orderVals.append(o)
#                     ratio.append(meanRP)
#
#     return subVals, diffVals, saVals, orderVals, ratio
#
# subject_list, difficulty_list, sa_list, order_list, ratio_list = makedf(0)
# preD = {'subject': subject_list, 'difficulty': difficulty_list, 'sa': sa_list, 'order': order_list, 'ratio': ratio_list}
# meanRatioPre = pd.DataFrame(preD)
#
# subject_list, difficulty_list, sa_list, order_list, ratio_list = makedf(1)
# d = {'subject': subject_list, 'difficulty': difficulty_list, 'sa': sa_list, 'order': order_list, 'ratio': ratio_list}
# meanRatioPost = pd.DataFrame(d)
#
# frames = [meanRatioPre, meanRatioPost]
# dataFrame = pd.concat(frames)