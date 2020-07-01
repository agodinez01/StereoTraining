import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pingouin as pg

withImage_anova = 'yes' # takes 'yes' or 'no' depending on whether code that runs the figure snd ANOVA should be executed.
run_type = 'CSF_acuity'     # takes 'CSF_acuity' or 'AULCSF' depending on the variable of interest. 'area' for area under the CSF curve and 'acuity' for CSF acuity

main_dir = "C:/Users/angie/Git Root/StereoTraining/data/"
results_dir = "C:/Users/angie/Git Root/StereoTraining/figs/cs_figs/"

# Load variables
os.chdir('C:/Users/angie/Git Root/StereoTraining/data')
cs_data = pd.read_csv('cs_data.csv')

subjects = cs_data.id.unique()
eye_tested = cs_data.condition.unique()
#
# sns.lmplot(x='test_num', y='AULCSF', hue='condition', col='group', data=cs_data)
# plt.ylim(0.5, 3)
# plt.yticks([1.0, 1.5, 2.0])
# plt.xlim(0.25, 6.25)
# plt.xticks([1, 2, 3, 4, 5, 6])
#
# plt.show()

def getRegressionCoeff(data):
    fit_params_linear = []
    slope = []
    condition = []

    for eye in eye_tested:
        x = data.test_num.unique().tolist()

        #linear fit
        if run_type == 'AULCSF':
            z_linear = np.polyfit(x, data.AULCSF[data.condition == eye], 1)
        elif run_type == 'CSF_acuity':
            z_linear = np.polyfit(x, data.CSF_acuity[data.condition == eye], 1)

        equation_linear = np.poly1d(z_linear)
        fit_params_linear.append(equation_linear)
        slope.append(z_linear[0])
        condition.append(eye)

    return fit_params_linear, slope, condition

def makeSlopeDF():
    subVals = []
    groupVals = []
    eyeVals = []
    slope = []

    for sub in subjects:
        data = cs_data.loc[cs_data.id == sub]
        group = data.group.tolist()[0]

        y, s, eye = getRegressionCoeff(data)

        subVals.append([sub, sub, sub])
        groupVals.append([group, group, group])
        slope.append(s)
        eyeVals.append(eye)

    return subVals, groupVals, slope, eyeVals

subList, groupList, slopeList, eyeList = makeSlopeDF()
sub_flat_list = [item for sublist in subList for item in sublist]
group_flat_list = [item for sublist in groupList for item in sublist]
eye_flat_list = [item for sublist in eyeList for item in sublist]
slope_flat_list = [item for sublist in slopeList for item in sublist]

frame = {'subject': sub_flat_list, 'group': group_flat_list, 'eye': eye_flat_list, 'slope': slope_flat_list}
dataFrame = pd.DataFrame(frame)

# Plot variables
plot_number = 1 # Start with first plot
plot_colors = [['#ff4f42', '#ff1100', '#ffb0ab'], ['#5980ff', '#0039f5', '#a1b7ff'], ['#666666', '#333333', '#999999'], ['#62d96d', '#009c0f', '#94fc8b']] # Red, blue, grey, green
plot_markers = ['o', 's', '^'] #DE, NDE, OU

if withImage_anova == 'yes':
    # 2-way ANOVA
    aov = pg.mixed_anova(dv='slope', between='group', within='eye', subject='subject', data=dataFrame)
    aov.round(3)
    aov

    # Bonferroni correction
    pvals = [aov['p-unc'][0], aov['p-unc'][1], aov['p-unc'][2]]
    reject, pvals_corr = pg.multicomp(pvals, method='fdr_bh')
    print(reject, pvals_corr)

    for sub in subjects:

        data = cs_data.loc[cs_data.id == sub]

        y, slope, eye = getRegressionCoeff(data)
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
        #sns.lmplot(x='test_num', y='AULCSF', hue='condition', data=data, palette=plot_palette, markers=plot_markers, size=14)

        if run_type == 'AULCSF':
            sns.scatterplot(x='test_num', y='AULCSF', hue='condition', data=data, palette=plot_palette, markers=plot_markers, style='condition', size=14)

            # Add slope to plot
            ax.text(3.5, 2.70, str('DE: ') + str(f"{slope[0]:.2f}"), fontsize=4, color=plot_palette[0])
            ax.text(3.5, 2.50, str('NDE: ') + str(f"{slope[1]:.2f}"), fontsize=4, color=plot_palette[1])
            ax.text(3.5, 2.30, str('OU: ') + str(f"{slope[2]:.2f}"), fontsize=4, color=plot_palette[2])
            ax.text(0.9, 2.40, str(sub), fontsize=6)

            # Set axes params
            # Set axes params
            plt.ylim(0.5, 3)
            plt.yticks([1.0, 1.5, 2.0])
            plt.xlim(0.25, 6.25)
            plt.xticks([1, 2, 3, 4, 5, 6])

            name = "csf_all_AULCSF.png"

        elif run_type == 'CSF_acuity':
            sns.scatterplot(x='test_num', y='CSF_acuity', hue='condition', data=data, palette=plot_palette,
                            markers=plot_markers, style='condition', size=14)

            # Add slope to plot
            ax.text(4.5, 8, str('DE: ') + str(f"{slope[0]:.2f}"), fontsize=4, color=plot_palette[0])
            ax.text(4.5, 5.5, str('NDE: ') + str(f"{slope[1]:.2f}"), fontsize=4, color=plot_palette[1])
            ax.text(4.5, 3, str('OU: ') + str(f"{slope[2]:.2f}"), fontsize=4, color=plot_palette[2])
            ax.text(0.9, 4, str(sub), fontsize=6)

            # Set axes params
            # Set axes params
            plt.ylim(0, 30)
            plt.yticks([10, 20])
            plt.xlim(0.25, 6.25)
            plt.xticks([1, 2, 3, 4, 5, 6])

            name = "csf_acuity.png"

        # Plot regression
        ax.plot(x, y[0](x), '--', color=plot_palette[0], linewidth=0.7)
        ax.plot(x, y[1](x), '--', color=plot_palette[1], linewidth=0.7)
        ax.plot(x, y[2](x), '--', color=plot_palette[2], linewidth=0.7)

        # Remove labels and legend
        ax.label_outer()
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.get_legend().remove()

        plot_number = plot_number + 1

    #plt.show()
    plt.savefig(fname=results_dir + name, bbox_inches='tight', format='png', dpi=300)