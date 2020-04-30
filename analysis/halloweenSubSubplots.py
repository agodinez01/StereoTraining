import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

#Run this if you have to create a new dataframe.
main_dir = "C:/Users/angie/Git Root/StereoTraining/GameObservers/"
results_dir = "C:/Users/angie/Git Root/StereoTraining/figs/hitRate/"

# Load variables
os.chdir('C:/Users/angie/Git Root/StereoTraining/data')
HallData = pd.read_csv('HalloweenSubjectData.csv')
hitRateData = pd.read_csv('HallHitRate4.csv')

# Define lists
subjects = hitRateData.subject.unique()
difficulty = hitRateData.difficulty.unique()
sa = hitRateData.sa.unique()
times = range(0,2)

def makedf():

    subVals = []
    conditionVals = []
    diffVals = []
    saVals = []
    ratioPreVals = []
    ratioPostVals = []
    groupVals = []

    for sub in subjects:
        subData = hitRateData.loc[hitRateData.subject == sub]

        for diff in difficulty:
            for item in sa:
                #for value in times:

                meanRatioPre = (subData.probabilityCorrect[(subData.difficulty == diff) & (subData.sa == item)][0:6]).mean()
                meanRatioPost = (subData.probabilityCorrect[(subData.difficulty == diff) & (subData.sa == item)][-6:]).mean()

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
                ratioPreVals.append(meanRatioPre)
                ratioPostVals.append(meanRatioPost)
                groupVals.append(group)

    return subVals, groupVals, conditionVals, diffVals, saVals, ratioPreVals, ratioPostVals

subject_list, group_list, condition_list, difficulty_list, sa_list, ratioPreVal_list, ratioPostVal_list = makedf()
frame = {'subject': subject_list, 'group': group_list, 'condition': condition_list, 'difficulty': difficulty_list, 'sa': sa_list, 'meanRatioPre': ratioPreVal_list, 'meanRatioPost': ratioPostVal_list}
dataF = pd.DataFrame(frame)

condition = hitRateData.condition.unique()

plotMarkers = ['o', 's', 'D']
plotColors = ['#595858', '#E91515', '#0923EF', '#0CB51F']  # Plot colors grey, red, blue and green

font = {'weight': 'bold', 'size': 18}
matplotlib.rc('font', **font)
sns.set('poster')

def makeAverageMatrix():
    condC = []
    diffC = []
    saC = []
    meanPreC = []
    meanPostC = []

    for cond in condition:
        for diff in difficulty:
            for item in sa:

                meanPre = dataF.meanRatioPre[(dataF.condition == cond) & (dataF.difficulty == diff) & (dataF.sa == item)].mean()
                meanPost = dataF.meanRatioPost[(dataF.condition == cond) & (dataF.difficulty == diff) & (dataF.sa == item)].mean()

                condC.append(cond)
                diffC.append(diff)
                saC.append(item)
                meanPreC.append(meanPre)
                meanPostC.append(meanPost)

    return condC, diffC, saC, meanPreC, meanPostC
condC, diffC, saC, meanPreC, meanPostC = makeAverageMatrix()
frame2 = {'condition': condC, 'difficulty': diffC, 'sa': saC, 'meanPreC': meanPreC, 'meanPostC': meanPostC}
df2 = pd.DataFrame(frame2)
df2.to_csv(r'C:\Users\angie\Git Root\StereoTraining\data\newHitRate.csv', index=False)

subplots = [221, 222, 223, 224]

def makeSubplots(data, s, fig):
    if s == 1000:
        ax1 = fig.add_subplot(221)
        plt.tick_params(labelbottom=False)
    elif s == 800:
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222, sharex=ax1, sharey=ax1)
        plt.tick_params(labelbottom=False, labelleft=False)
    elif s == 600:
        ax1 = fig.add_subplot(221)
        ax3 = fig.add_subplot(223, sharex=ax1, sharey=ax1)
    elif s == 400:
        ax1 = fig.add_subplot(221)
        ax4 = fig.add_subplot(224, sharex=ax1, sharey=ax1)
        plt.tick_params(labelleft=False)

    for cond in condition:
        if cond == 'binocular':
            markerFill = ['', 'none', plotColors[0], 'none']
            plotColor = plotColors[0]
        elif cond == 'strabismus':
            markerFill = ['', 'none', plotColors[1], 'none']
            plotColor = plotColors[1]
        elif cond == 'anisometropia':
            markerFill = ['', 'none', plotColors[2], 'none']
            plotColor = plotColors[2]
        elif cond == 'stereo-weak':
            markerFill = ['', 'none', plotColors[3], 'none']
            plotColor = plotColors[3]

        sns.scatterplot(x='meanPreC', y='meanPostC', data=data[(data.sa == s) & (data.condition == cond)], style='difficulty',
                        color=plotColor, markers=plotMarkers, legend=False, alpha=0.75)

    plt.plot([0.8, 1.0], [0.8, 1.0], 'k--')
    plt.text(0.96, 0.81, s, fontsize=14)
    plt.yticks([0.8, 0.9, 1])
    plt.xticks([0.8, 0.9, 1])
    plt.xlabel('')
    plt.ylabel('')

    for ax in fig.get_axes():
        ax.label_outer()


def make2x2AllFig(data):
    #Make big figure to encompass subplots
    fig = plt.figure()
    sns.set_style('white')

    ax = fig.add_subplot(111)
    ax.spines['left'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
    plt.xlabel('Pre-game proportion correct')
    plt.ylabel('Post-game proportion correct')

    for item in [0, 1, 2, 3]:
        plt.plot([], label=item, linestyle='-', color=plotColors[item])

    for diff in [0, 1, 2]:
        plt.scatter([], [], c='#595858', marker=plotMarkers[diff], label=diff)

    plt.legend()
    L = plt.legend(bbox_to_anchor=(1.01, 1.01), loc=2, borderaxespad=0, prop={'size': 13}, frameon=False)
    L.get_texts()[0].set_text('Binocular')
    L.get_texts()[1].set_text('Strabismus')
    L.get_texts()[2].set_text('Anisometropia')
    L.get_texts()[3].set_text('Stereo-weak')
    L.get_texts()[4].set_text('Level 1: All cues')
    L.get_texts()[5].set_text('Level 2: Motion parallax & disparity')
    L.get_texts()[6].set_text('Level 3: Disparity only')

    for s in sa:
        makeSubplots(data, s, fig)

    #plt.show()
    #data
    plt.savefig(fname=results_dir + 'AverageData.png', bbox_inches='tight', format='png', dpi=300)

make2x2AllFig(df2)

def make2x2PanelPlot(subData, condition):
    fig = plt.figure()
    sns.set_style('white')

    if condition == 'binocular':
        markerFill = ['none', plotColors[0], 'none']
        plotColor = plotColors[0]
    elif condition == 'strabismus':
        markerFill = ['none', plotColors[1], 'none']
        plotColor = plotColors[1]
    elif condition == 'anisometropia':
        markerFill = ['none', plotColors[2], 'none']
        plotColor = plotColors[2]
    elif condition == 'stereo-weak':
        markerFill = ['none', plotColors[3], 'none']
        plotColor = plotColors[3]

    ax = fig.add_subplot(111)
    ax.spines['left'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
    plt.xlabel('Pre-game proportion correct')
    plt.ylabel('Post-game proportion correct')
    plt.title(sub)

    sns.set_style('whitegrid')
    ax1 = fig.add_subplot(221)
    plt.plot([0.7, 1.0], [0.7, 1.0], 'k--')
    plt.text(0.95, 0.71, '400"', fontsize=14)
    sns.scatterplot(x='meanRatioPre', y='meanRatioPost', data=subData[subData.sa == 400], style='difficulty',
                    markers=plotMarkers, legend=False, color=plotColor)
    plt.xlabel('')
    plt.ylabel('')

    ax2 = fig.add_subplot(222, sharex=ax1, sharey=ax1)
    plt.plot([0.7, 1.0], [0.7, 1.0], 'k--')
    plt.text(0.95, 0.71, '600"', fontsize=14)
    sns.scatterplot(x='meanRatioPre', y='meanRatioPost', data=subData[subData.sa == 600], style='difficulty',
                    markers=plotMarkers, legend='full', color=plotColor)
    plt.xlabel('')
    plt.ylabel('')
    L = plt.legend(bbox_to_anchor=(1.01, 1.01), loc=2, borderaxespad=0, prop={'size': 10}, frameon=False)
    L.get_texts()[0].set_text('Difficulty')
    L.get_texts()[1].set_text('Level 1: All cues')
    L.get_texts()[2].set_text('Level 2: Motion parallax & disparity')
    L.get_texts()[3].set_text('Level 3: Disparity only')

    ax3 = fig.add_subplot(223, sharex=ax1, sharey=ax1)
    plt.plot([0.7, 1.0], [0.7, 1.0], 'k--')
    plt.text(0.95, 0.71, '800"', fontsize=14)
    sns.scatterplot(x='meanRatioPre', y='meanRatioPost', data=subData[subData.sa == 800], style='difficulty',
                    markers=plotMarkers, legend=False, color=plotColor)
    plt.xlabel('')
    plt.ylabel('')

    ax4 = fig.add_subplot(224, sharex=ax1, sharey=ax1)
    plt.plot([0.7, 1.0], [0.7, 1.0], 'k--')
    plt.text(0.95, 0.71, '1000"', fontsize=14)
    sns.scatterplot(x='meanRatioPre', y='meanRatioPost', data=subData[subData.sa == 1000], style='difficulty',
                    markers=plotMarkers, legend=False, color=plotColor)
    plt.xlabel('')
    plt.ylabel('')

    for ax in fig.get_axes():
        ax.label_outer()

    name = subData.subject.tolist()[0] + 'hitRate.png'
    #plt.show()
    #plt.savefig(fname= results_dir + name, bbox_inches= 'tight', format= 'png', dpi= 300)

for sub in subjects:
    subData = dataF.loc[dataF.subject == sub]
    condition = subData.condition.tolist()[0]
    make2x2PanelPlot(subData, condition)


# def make2x2AllFig(data):
#     fig = plt.figure()
#     sns.set_style('white')
#
#     ax = fig.add_subplot(111)
#     ax.spines['left'].set_color('none')
#     ax.spines['bottom'].set_color('none')
#     ax.spines['right'].set_color('none')
#     ax.spines['top'].set_color('none')
#     ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
#     plt.xlabel('Pre-game proportion correct')
#     plt.ylabel('Post-game proportion correct')
#
#     for cond in condition:
#
#         if cond == 'binocular':
#             markerFill = ['', 'none', plotColors[0], 'none']
#             plotColor = plotColors[0]
#         elif cond == 'strabismus':
#             markerFill = ['', 'none', plotColors[1], 'none']
#             plotColor = plotColors[1]
#         elif cond == 'anisometropia':
#             markerFill = ['', 'none', plotColors[2], 'none']
#             plotColor = plotColors[2]
#         elif cond == 'stereo-weak':
#             markerFill = ['', 'none', plotColors[3], 'none']
#             plotColor = plotColors[3]
#
#         sns.set_style('whitegrid')
#
#         ax1 = fig.add_subplot(221)
#         plt.plot([0.8, 1.0], [0.8, 1.0], 'k--')
#         plt.text(0.97, 0.81, '1000"', fontsize=14)
#         for diff in difficulty:
#             sns.scatterplot(x='meanPreC', y='meanPostC', data=data[(data.sa == 1000) & (data.condition == cond)], style='difficulty',
#                             markers=plotMarkers, facecolor=markerFill[diff], legend=False, color=plotColor, alpha=0.75)
#         plt.show()
#         plt.yticks([0.8, 0.9, 1])
#         plt.xticks([0.8, 0.9, 1])
#         plt.tick_params(labelbottom='off')
#         plt.xlabel('')
#         plt.ylabel('')
#
#         ax2 = fig.add_subplot(222, sharex=ax1, sharey=ax1)
#         plt.plot([0.8, 1.0], [0.8, 1.0], 'k--')
#         plt.text(0.97, 0.81, '800"', fontsize=14)
#         sns.scatterplot(x='meanPreC', y='meanPostC', data=data[(data.sa == 800) & (data.condition == cond)], style='difficulty',
#                         markers=plotMarkers, legend=False, color=plotColor, alpha=0.75)
#         plt.tick_params(labelbottom='off', labelleft='off')
#         plt.xlabel('')
#         plt.ylabel('')
#
#         ax3 = fig.add_subplot(223, sharex=ax1, sharey=ax1)
#         plt.plot([0.8, 1.0], [0.8, 1.0], 'k--')
#         plt.text(0.97, 0.81, '600"', fontsize=14)
#         sns.scatterplot(x='meanPreC', y='meanPostC', data=data[(data.sa == 600) & (data.condition == cond)], style='difficulty',
#                         markers=plotMarkers, legend=False, color=plotColor, alpha=0.75)
#         plt.xlabel('')
#         plt.ylabel('')
#
#         ax4 = fig.add_subplot(224, sharex=ax1, sharey=ax1)
#         plt.plot([0.8, 1.0], [0.8, 1.0], 'k--')
#         plt.text(0.97, 0.81, '400"', fontsize=14)
#         sns.scatterplot(x='meanPreC', y='meanPostC', data=data[(data.sa == 400) & (data.condition == cond)], style='difficulty',
#                         markers=plotMarkers, legend=False, color=plotColor, alpha=0.75)
#         plt.tick_params(labelleft='off')
#         plt.xlabel('')
#         plt.ylabel('')
#
#     plt.show()
#     #plt.savefig(fname=results_dir + 'AverageData.png', bbox_inches='tight', format='png', dpi=300)
#
# make2x2AllFig(df2)