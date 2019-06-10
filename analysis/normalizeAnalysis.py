# This script takes the csv file produced by buildDataFrame to normalize the data.
# TODO: Get slope from normalized data to assess "learning"/improvement in-game
# TODO: Make Learning Curve plots for each subject.

import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import scipy.stats

main_dir = "C:/Users/angie/Git Root/StereoTraining/GameObservers/"
sub_dir = "/DartBoard/"
results_dir = "C:/Users/angie/Git Root/StereoTraining/figs/"
Control = {'ah', 'aj', 'dd', 'dl', 'ez', 'it', 'll', 'sh', 'sm', 'sr'} #10 control
Anomalous = {'bb', 'by', 'co', 'et', 'gn', 'gp', 'jz', 'kp', 'ky', 'mb', 'mg', 'ni', 'tp'}  #13 experimental
obs_set = {'ah', 'aj', 'dd', 'dl', 'ez', 'it', 'll', 'sh', 'sm', 'sr', 'bb', 'by', 'co', 'et', 'gn', 'gp', 'jz', 'kp', 'ky', 'mb', 'mg', 'ni', 'tp'}

# Load variables
os.chdir('C:/Users/angie/Git Root/StereoTraining/data')
allData = pd.read_csv('subjectData.csv')

subjects = allData.subject.unique()
difficulties = allData.difficulty.unique()

# For each subject, date, and difficulty level we get median "in-game" stereoacuity of 50 arc secs or more.
def getSa():
    saMed = []
    saSub = []
    saDate = []
    saDiff = []
    saSEM = []
    saLog = []
    saLogSEM = []
    saVar = []

    for sub in subjects:
        subData = allData.loc[allData.subject == sub]
        subData = subData.loc[subData['stereoacuity'] > 50]
        dates = subData.date.unique()

        for date in dates:
            for difficulty in difficulties:
                subjectSaMedian = subData.stereoacuity[(subData.date == date) & (subData.difficulty == difficulty)].median()
                subjectSaVar = np.var(np.log10(subData.stereoacuity[(subData.date == date) & (subData.difficulty == difficulty)]))
                subjectSEM = subData.stereoacuity[(subData.date == date) & (subData.difficulty == difficulty)].sem()
                subjectLog = np.log10(subjectSaMedian)
                subjectSEMLog = np.log10(subjectSEM)

                saSub.append(sub)
                saDate.append(date)
                saDiff.append(difficulty)

                saMed.append(subjectSaMedian)
                saSEM.append(subjectSEM)
                saLog.append(subjectLog)
                saLogSEM.append(subjectSEMLog)
                saVar.append(subjectSaVar)

    return  saSub, saDate, saDiff, saMed, saSEM, saLog, saLogSEM, saVar

subject_list, date_list, difficulty_list, saMed_list, saSEM_list, saLog_list, saLogSEM_list, saVar_list = getSa()
d = {'subject': subject_list, 'date': date_list, 'difficulty': difficulty_list, 'saMedian': saMed_list, 'saSEM': saSEM_list, 'saLog': saLog_list, 'saLogSEM': saLogSEM_list, 'saVar': saVar_list}
df = pd.DataFrame(d)
df.to_csv(r'C:\Users\angie\Git Root\StereoTraining\data\dataFrame.csv', index=False)

# Function calculates the term that will be used to normalize the data.
# For each subject and difficulty, we take the log mean of the last 5 sessions (10 blocks).
def getNormalizer():
    normTerm = []
    normSub = []
    normDiff = []

    for sub in subjects:
        for diff in difficulties:
            nTerm = df.saLog[(df.subject == sub) & (df.difficulty == diff)][-10:].mean()
            normTerm.append(nTerm)

            normSub.append(sub)
            normDiff.append(diff)
    return normTerm, normSub, normDiff

normTerm, normSub, normDiff = getNormalizer()

normTable = {'subject': normSub, 'difficulty': normDiff, 'normalizer': normTerm}
normDf = pd.DataFrame(normTable)

# TODO: Make sure this works... currently I dont think it does.
# Function takes the value from getNormalizer function and subtracts it from the log mean value from each observer, date, and difficulty level.
def normalizeData():
    normalized = []
    subject = []
    difficultyList = []
    dateList = []
    normSEM = []
    normalizedVariance = []

    for sub in subjects:
        for difficulty in difficulties:

            normalizedVals = (np.subtract(df.saLog[(df.subject == sub) & (df.difficulty == difficulty)], normDf.normalizer[(normDf.subject == sub) & (normDf.difficulty == difficulty)])).tolist()
            normalized.append(normalizedVals)

            subject1 = [sub] * len(normalizedVals)
            subject.append(subject1)

            difficulty1 = [difficulty] * len(normalizedVals)
            difficultyList.append(difficulty1)

            dateL = (df.date[(df.subject == sub) & (df.difficulty == difficulty)]).tolist()
            dateList.append(dateL)

    return normalized, subject, difficultyList, dateList, normSEM

normalizedVals, subjectList, difficultyList, dateList, semList = normalizeData()

normFlatList = [item for sublist in normalizedVals for item in sublist]
subFlatList = [item for sublist in subjectList for item in sublist]
diffFlatList = [item for sublist in difficultyList for item in sublist]
dateFlatList = [item for sublist in dateList for item in sublist]
#semFlatList = [item for sublist in semList for item in sublist]

normDataTable = {'subject': subFlatList, 'date': dateFlatList, 'difficulty': diffFlatList, 'normalizedVals': normFlatList, 'normalizedSEM': df.saLogSEM}
dataFrame = pd.DataFrame(normDataTable)
dataFrame.to_csv(r'C:\Users\angie\Git Root\StereoTraining\data\normalizedDataFrame.csv', index=False)

for sub in subjects:
    # Fit data
    fit_params_poly = []
    fit_params_linear = []
    x = []
    chi_squared_poly = []
    chi_squared_linear = []

    for diff in difficulties:
        x = np.arange(0, len(dataFrame.loc[(dataFrame['subject'] == sub) & (dataFrame['difficulty'] == diff)]['date'].unique()), step=1)

        #Fit polynomial
        z_poly = np.polyfit(x, dataFrame.loc[(dataFrame['subject'] == sub) & (dataFrame['difficulty'] == diff)]['normalizedVals'], 2)
        equation_poly = np.poly1d(z_poly)

        #Linear fit
        z_linear = np.polyfit(x, dataFrame.loc[(dataFrame['subject'] == sub) & (dataFrame['difficulty'] == diff)][
            'normalizedVals'], 1)
        equation_linear = np.poly1d(z_linear)

        #Reduced Chi-squared (poly)
        chi_s = np.sum(((np.subtract(dataFrame.loc[(dataFrame['subject'] == sub) & (dataFrame['difficulty'] == diff)]['normalizedVals'],
                              np.polyval(equation_poly, x)) ** 2) / df.saVar[(df['subject'] == sub) & (df['difficulty'] == diff)]) / 2)

        chi_squared_poly.append('%.3f'%chi_s)

        #Reduced chi-squared (linear)
        chi_s_linear = np.sum(((np.subtract(dataFrame.loc[(dataFrame['subject'] == sub) & (dataFrame['difficulty'] == diff)]['normalizedVals'],
                              np.polyval(equation_linear, x)) ** 2) / df.saVar[(df['subject'] == sub) & (df['difficulty'] == diff)]) / 1)

        chi_squared_linear.append('%.3f'%chi_s_linear)

        fit_params_poly.append(equation_poly)
        fit_params_linear.append(equation_linear)

    # Plot params
    font = {'weight': 'bold', 'size': 18}
    matplotlib.rc('font', **font)
    sns.set('poster', palette='colorblind')
    sns.set_style('whitegrid')

    plt.title("Learning curve: " + sub.upper())

    ax = sns.scatterplot(x='date', y='normalizedVals', data=dataFrame.loc[dataFrame['subject'] == sub],
                          hue='difficulty', palette=sns.color_palette('colorblind', n_colors=3), alpha=.75)

    ax.plot(x, fit_params_poly[0](x), '-')
    ax.plot(x, fit_params_poly[1](x), '-')
    ax.plot(x, fit_params_poly[2](x), '-')
    ax.plot(x, fit_params_linear[0](x), '--', color='#33ccff')
    ax.plot(x, fit_params_linear[1](x), '--', color='#F0E68C')
    ax.plot(x, fit_params_linear[2](x), '--', color='#66CDAA')

    #TODO: Need to get the right SEM.. currently, errorbars are huge.. need to normalize
    ax.set(xticklabels=np.arange(1, len(df.loc[df['subject'] == sub].date.unique()), step=10), xlabel='Session number',
            ylabel='Normalized')
    plt.yticks(np.arange(-0.5, 1, step=0.5))
    plt.xticks(np.arange(0, len(dataFrame.loc[df['subject'] == sub].date.unique()), step=10))

    L = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='xx-small')
    L.get_texts()[0].set_text('Difficulty')
    L.get_texts()[1].set_text('Natural')
    L.get_texts()[2].set_text('Advanced')
    L.get_texts()[3].set_text('Expert')

    plt.text(0.9, -0.35, 'Chi sq: ' + chi_squared_poly[0], fontsize=8, color='#0086b3')
    plt.text(12, -0.35, 'Chi sq Linear: ' + chi_squared_linear[0], fontsize=8, color='#33ccff')
    plt.text(0.9, -0.4, 'Chi sq: ' + chi_squared_poly[1], fontsize=8, color='#ffbf00')
    plt.text(12, -0.4, 'Chi sq Linear: ' + chi_squared_linear[1], fontsize=8, color='#F0E68C')
    plt.text(0.9, -0.45, 'Chi sq: ' + chi_squared_poly[2], fontsize=8, color='#008000')
    plt.text(12, -0.45, 'Chi sq Linear: ' + chi_squared_linear[2], fontsize=8, color='#66CDAA')

    name = sub + "_NormalizedLearningCurve.png"
    plt.savefig(fname=results_dir + name, bbox_inches='tight', format='png', dpi=300)
    plt.clf()

# for sub in subjects:
#     # Plot params
#     font = {'weight': 'bold', 'size': 18}
#     matplotlib.rc('font', **font)
#     sns.set('poster', palette='colorblind')
#     sns.set_style('whitegrid')
#
#     # Top panel
#     plt.figure(1)
#     plt.subplot(2, 1, 1)
#
#     ax = sns.scatterplot(x='date', y='saMedian', data=df.loc[df['subject'] == sub], hue='difficulty', palette=sns.color_palette('colorblind', n_colors=3))
#     plt.errorbar(x='date', y='saMedian', data=df.loc[df['subject'] == sub], yerr='saSEM', fmt='none', hue='difficulty')
#
#     ax.set(xticklabels=np.arange(1, len(df.loc[df['subject'] == sub].date.unique()), step=10), yticklabels=np.arange(50, 850, step=100), ylabel='Stereoacuity (arc secs)', xlabel=None)
#     plt.yticks(np.arange(50, 850, step=150))
#     plt.xticks(np.arange(0, len(df.loc[df['subject'] == sub].date.unique()), step=10))
#     plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='xx-small')
#     plt.title("Learning curve: " + sub.upper())
#
#     # Bottom panel
#     plt.subplot(2, 1, 2)
#     ax1 = sns.scatterplot(x='date', y='normalizedVals', data=dataFrame.loc[dataFrame['subject'] == sub], hue='difficulty', palette=sns.color_palette('colorblind', n_colors=3), legend=False)
#     ax1.set(xticklabels=np.arange(1, len(df.loc[df['subject'] == sub].date.unique()), step=10), xlabel='Session number', ylabel='Normalized')
#     plt.yticks(np.arange(-0.5, 1, step=.5))
#     plt.xticks(np.arange(0, len(dataFrame.loc[df['subject'] == sub].date.unique()), step=10))
#
#     plt.show()
#
#     name = sub+"_learningCurve.png"
#     plt.savefig(fname=results_dir + name, bbox_inches='tight', format='png', dpi=1000)
#     plt.clf()
#
# def getHit(var1, var2):
#     result = []
#     for sub in obs_set:
#         hitRate = var1.loc[sub, [1, 2, 3], True] / var2[sub]
#         result.append(hitRate)
#     return result
#
# hitRateBySub = getHit(subjectHit, subjectHitTotal)
#
# dataSavePath = main_dir + 'processed_data.pkl'
# allData.to_pickle(dataSavePath)
#
# ## SA performance
# plt.figure()
# # sns.set_context("poster")
# ax = sns.boxplot(x='difficulty', y='stereoacuity', data=allData, hue='group', palette='colorblind', fliersize=0)
# ax.set(ylabel='Stereoacuity (arc secs)', xticklabels=['Natural', 'Advanced', 'Expert'])
# ax.set_ylim(bottom=None, top=1500)
# plt.legend(loc=1)
#
# plt.show()
#
# name = "Effect of group and difficulty level on stereoacuity.jpeg"
# plt.savefig(fname=results_dir + name, bbox_inches='tight', format='pdf', dpi=1000)

# def func(x, a, b, c):
#     return a * np.exp(-b * x) +c

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
