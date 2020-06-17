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
plot_colors = [['#ff4f42', '#ffb0ab', '#ff1100'], ['#5980ff', '#a1b7ff', '#0039f5'], ['#666666', '#999999', '#333333'], ['#62d96d', '#bdffc3', '#009c0f']]

for sub in subjects:
    data = cs_data.loc[cs_data.id == sub]
    y = getRegressionCoeff(data)
    x = data.test_num.unique().tolist()

    # Assign color for plots
    if sub[0:3] == 'ASW':
        plot_palette = plot_colors[3]
    elif sub[0:2] == 'AS' or sub[0:2] == 'AM':
        plot_palette = plot_colors[0]
    elif sub[0:2] == 'AA':
        plot_palette = plot_colors[1]
    elif sub[0] == 'N':
        plot_palette = plot_colors[2]

    ax = plt.subplot(5, 4, plot_number)
    sns.scatterplot(x='test_num', y='AULCSF', hue='condition', data=data, palette=plot_palette)

    # Plot regression
    ax.plot(x, y[0](x), '--', color=plot_palette[0])
    ax.plot(x, y[1](x), '--', color=plot_palette[1])
    ax.plot(x, y[2](x), '--', color=plot_palette[2])

    # Add slope to plot
    ax.text(3.5, 2.70, str(y[0]), fontsize=4, color=plot_palette[0])
    ax.text(3.5, 2.50, str(y[1]), fontsize=4, color=plot_palette[1])
    ax.text(3.5, 2.30, str(y[2]), fontsize=4, color=plot_palette[2])
    ax.text(0.9, 2.40, str(sub), fontsize=6)

    # Set axes params
    plt.ylim(0.5, 3)
    plt.yticks([1.0, 1.5, 2.0])
    plt.xlim(0.25, 6.25)
    plt.xticks([1, 2, 3, 4, 5, 6])

    # Remove labels and legend
    ax.label_outer()
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.get_legend().remove()

    plot_number = plot_number + 1

name = "csf_all.png"
plt.savefig(fname=results_dir + name, bbox_inches='tight', format='png', dpi=300)
