import pandas as pd
import os
import math
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import pingouin as pg

main_dir = "C:/Users/angie/Git Root/StereoTraining/data/"
results_dir = "C:/Users/angie/Git Root/StereoTraining/figs/cs_figs/"

# Load variables
os.chdir('C:/Users/angie/Git Root/StereoTraining/data')
cs_data = pd.read_csv('cs_data.csv')

subjects = cs_data.id.unique()
eye_tested = cs_data.condition.unique()


def getRegressionCoeff(data):
    #eye_cond = []
    fit_params_linear = []

    for eye in eye_tested:
        x = data.test_num.unique().tolist()

        #linear fit
        z_linear = np.polyfit(x, data.AULCSF[data.condition == eye], 1)
        equation_linear = np.poly1d(z_linear)
        fit_params_linear.append(equation_linear)

    return fit_params_linear

# Plot variables
plot_number = 1 # Start with first plot
plot_colors = []

for sub in subjects:
    data = cs_data.loc[cs_data.id == sub]
    y = getRegressionCoeff(data)
    x = data.test_num.unique().tolist()

    ax = plt.subplot(5, 4, plot_number)
    sns.scatterplot(x='test_num', y='AULCSF', hue='condition', data=data)
    ax.plot(x, y[0](x), '--')
    ax.plot(x, y[1](x), '--')
    ax.plot(x, y[2](x), '--')

    # Set axes params
    plt.ylim(0.5, 2.5)
    plt.yticks([1.0, 1.5, 2.0])

    # Remove labels and legend
    ax.label_outer()
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.get_legend().remove()

    plot_number = plot_number + 1


# def getRegression(data):
#     fit_params_linear = []
#     x = []
#     #chi_square_linear = []
#
#     for cond in eye_tested:
#         x = np.arange(0, len(data.runs.unique()), step=1)
#
#         # Linear fit
#         z_linear = np.polyfit(x, data.loc[data.condition == cond]['AULCSF'], 1)
#         equation_linear = np.polyld(z_linear)
#
#         fit_params_linear.append(z_linear)
#         x.append(x)
#
#         # # Reduced chi-squared
#         # chi_s = np.sum(((np.subtract)))



plt.show()
cs_data