import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

main_dir = "C:/Users/angie/Git Root/StereoTraining/GameObservers/"
sub_dir = "/DartBoard/"
results_dir = "C:/Users/angie/Git Root/StereoTraining/figs/LearningRaw/"
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
    saLog = []
    saVar = []
    saLogVar = []

    for sub in subjects:
        subData = allData.loc[allData.subject == sub]
        subData = subData.loc[subData['stereoacuity'] > 50]
        dates = subData.date.unique()

        for date in dates:
            for difficulty in difficulties:
                subjectSaMedian = subData.stereoacuity[(subData.date == date) & (subData.difficulty == difficulty)].median()
                subjectSaVar = np.var(subData.stereoacuity[(subData.date == date) & (subData.difficulty == difficulty)])
                subjectSaLogVar = np.var(np.log10(subData.stereoacuity[(subData.date == date) & (subData.difficulty == difficulty)]))
                subjectLog = np.log10(subjectSaMedian)

                saSub.append(sub)
                saDate.append(date)
                saDiff.append(difficulty)

                saMed.append(subjectSaMedian)
                saLog.append(subjectLog)
                saVar.append(subjectSaVar)
                saLogVar.append(subjectSaLogVar)

    return saSub, saDate, saDiff, saMed, saVar, saLog, saLogVar

subject_list, date_list, difficulty_list, saMed_list, saVar_list, saLog_list, saLogVar_list  = getSa()
d = {'subject': subject_list, 'date': date_list, 'difficulty': difficulty_list, 'saMedian': saMed_list, 'saVar': saVar_list, 'saLog': saLog_list, 'saLogVar': saLogVar_list}
df = pd.DataFrame(d)
df.to_csv(r'C:\Users\angie\Git Root\StereoTraining\data\dataFrame.csv', index=False)

# TODO: Make into function

for sub in subjects:
    fit_params_poly = []
    fit_params_linear = []
    x = []
    chi_squared_poly = []
    chi_squared_linear = []

    for diff in difficulties:
        x = np.arange(0, len(df.loc[(df['subject'] == sub) & (df['difficulty'] == diff)]['date'].unique()), step=1)

        # Fit polynomial
        z_poly = np.polyfit(x, df.loc[(df['subject'] == sub) & (df['difficulty'] == diff)][
            'saMedian'], 2)
        equation_poly = np.poly1d(z_poly)

        # Linear fit
        z_linear = np.polyfit(x, df.loc[(df['subject'] == sub) & (df['difficulty'] == diff)][
            'saMedian'], 1)
        equation_linear = np.poly1d(z_linear)

        # Reduced Chi-squared (poly)
        chi_s = np.sum(((np.subtract(
            df.loc[(df['subject'] == sub) & (df['difficulty'] == diff)]['saMedian'],
            np.polyval(equation_poly, x)) ** 2) / df.saVar[
                            (df['subject'] == sub) & (df['difficulty'] == diff)]) / 2)

        chi_squared_poly.append('%.3f' % chi_s)

        # Reduced chi-squared (linear)
        chi_s_linear = np.sum(((np.subtract(
            df.loc[(df['subject'] == sub) & (df['difficulty'] == diff)]['saMedian'],
            np.polyval(equation_linear, x)) ** 2) / df.saVar[
                                   (df['subject'] == sub) & (df['difficulty'] == diff)]) / 1)

        chi_squared_linear.append('%.3f' % chi_s_linear)

        fit_params_poly.append(equation_poly)
        fit_params_linear.append(equation_linear)

    # Plot params
    font = {'weight': 'bold', 'size': 18}
    matplotlib.rc('font', **font)
    sns.set('poster', palette='colorblind')
    sns.set_style('whitegrid')

    ax = sns.scatterplot(x='date', y='saMedian', data=df.loc[df['subject'] == sub],
                         hue='difficulty', palette=sns.color_palette('colorblind', n_colors=3), alpha=.85)
    ax.set_yscale('log')
    ax.set_ylim(50, 850)
    ax.set_yticks([100, 300, 500, 800])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    ax.plot(x, fit_params_poly[0](x), '-')
    ax.plot(x, fit_params_poly[1](x), '-')
    ax.plot(x, fit_params_poly[2](x), '-')
    ax.plot(x, fit_params_linear[0](x), '--', color='#33ccff')
    ax.plot(x, fit_params_linear[1](x), '--', color='#F0E68C')
    ax.plot(x, fit_params_linear[2](x), '--', color='#66CDAA')

    ax.set(xticklabels=np.arange(1, len(df.loc[df['subject'] == sub].date.unique()), step=10),
            xlabel='Session number', ylabel='In-game stereoacuity (arc secs)')
    #plt.yticks(np.arange(-0.5, 1, step=0.5))
    #plt.ylim(50, 850)
    #plt.yticks(np.arange(50, 850, step=150))
    plt.xticks(np.arange(0, len(df.loc[df['subject'] == sub].date.unique()), step=10))
    plt.title("Learning curve: " + sub.upper())

    L = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='xx-small')
    L.get_texts()[0].set_text('Difficulty')
    L.get_texts()[1].set_text('Natural')
    L.get_texts()[2].set_text('Advanced')
    L.get_texts()[3].set_text('Expert')

    plt.text(0.9, 90, 'Chi sq: ' + chi_squared_poly[0], fontsize=8, color='#0086b3')
    plt.text(12, 90, 'Chi sq Linear: ' + chi_squared_linear[0], fontsize=8, color='#33ccff')
    plt.text(0.9, 80, 'Chi sq: ' + chi_squared_poly[1], fontsize=8, color='#ffbf00')
    plt.text(12, 80, 'Chi sq Linear: ' + chi_squared_linear[1], fontsize=8, color='#F0E68C')
    plt.text(0.9, 70, 'Chi sq: ' + chi_squared_poly[2], fontsize=8, color='#008000')
    plt.text(12, 70, 'Chi sq Linear: ' + chi_squared_linear[2], fontsize=8, color='#66CDAA')

    name = sub + "_LearningCurve.png"
    plt.savefig(fname=results_dir + name, bbox_inches='tight', format='png', dpi=300)
    plt.clf()
