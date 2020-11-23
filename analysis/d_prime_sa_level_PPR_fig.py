import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

main_dir = "C:/Users/angie/Git Root/StereoTraining/data"
results_dir = "C:/Users/angie/Git Root/StereoTraining/figs/"

# Load variables
os.chdir('C:/Users/angie/Git Root/StereoTraining/data')
dData = pd.read_csv('d_prime_PPR_by_level_and_demand.csv')

dData2 = pd.read_csv('d_prime_santiago.csv')

subjects = dData2.id.unique()
levels = dData2.level.unique()
stereoDemand = dData2.sa.unique()
groups = dData2.group.unique()

def get_subject_PPR():
    subjectIDVal = []
    conditionVal = []
    groupVal = []
    levelVal = []
    saVal = []
    sa_s = []

    PPR = []

    for sub in subjects:
        for level in levels:
            for sa in stereoDemand:

                PPR_indv = dData2['d prime post'][(dData2.id == sub) & (dData2.level == level) & (dData2.sa == sa)].mean() / dData2['d prime pre'][(dData2.id == sub) & (dData2.level == level) & (dData2.sa == sa)].mean()

                if sub[0] == 'N':
                    g = 'normal'
                    sas = sa - 5
                    cond = 'normal'
                else:
                    g = 'anomalous'
                    sas = sa + 5

                    if sub[1] == 'A':
                        cond = 'anisometropia'
                    elif sub[2] == 'W':
                        cond = 'stereo-weak'
                    else:
                        cond = 'strabismus'

                subjectIDVal.append(sub)
                levelVal.append(level)
                saVal.append(sa)
                sa_s.append(sas)
                groupVal.append(g)
                PPR.append(PPR_indv)
                conditionVal.append(cond)
    return subjectIDVal, groupVal, conditionVal, levelVal, saVal, sa_s, PPR

subject_list, group_list, condition_list, level_list, sa_list, sas_list, PPR_list = get_subject_PPR()
d = {'subject': subject_list, 'group': group_list, 'condition': condition_list, 'level': level_list, 'sa': sa_list, 'saN': sas_list, 'PPR': PPR_list}
df = pd.DataFrame(d)

plt.clf()


font = {'weight': 'normal', 'size': 16}
matplotlib.rc('font', **font)
#sns.set('poster')
sns.set_style('white')

fig = plt.figure(figsize=(12,5))
plt.xlabel('Stereoacuity demand (arc secs)')

#start at plot panel 1
plot_number = 1
plotColors = ['#6e6e6e', '#cccccc']
condColors = {'normal': '#000000', 'anisometropia':'#0651d1', 'strabismus':'#d10606', 'stereo-weak': '#12cf04'}
for level in levels:
    ax = plt.subplot(1, 3, plot_number)

    #actual plot
    #sns.boxplot(x='sa', y='PPR', hue='group', data=df[df.level == level], showfliers=False, palette=plotColors)
    sns.boxplot(x='sa', y='PPR', hue='group', data=df[df.level == level], showfliers=False, palette=plotColors)
    #plt.scatter(x=df.sa[(df.level == level) & (df.group == 'normal')], y=df.PPR[(df.level == level) & (df.group == 'normal')])
    #sns.stripplot(x='sa', y='PPR', hue='condition', data=df[df.level == level], jitter=True, dodge=True, palette=condColors, alpha=0.8)

    sns.stripplot(x='sa', y='PPR', hue='group', data=df[df.level == level], jitter=True, dodge=True, color='grey', alpha=0.8)

    #sns.swarmplot(x='sa', y='PPR', hue='group', data=df[df.level == level])

    plt.axhline(y=1, color='k', Linestyle='--')

    #set axes params
    plt.ylim(0, 4)
    plt.yticks([1, 2, 3, 4])

    #remove labels and legend
    ax.label_outer()
    ax.set_xlabel('')
    ax.get_legend().remove()

    if plot_number == 1:
        ax.set_ylabel('PPR (post:pre ratio)', fontweight='bold')
    elif plot_number == 2:
        ax.set_xlabel('Stereoacuity (arc secs)', fontweight='bold')
    else:
        ax.set_ylabel('')

    #ax.set_title('Level ' + str(plot_number))

    #Increase the plot number before going through the loop again
    plot_number = plot_number + 1


L = plt.legend(bbox_to_anchor=(1, 0.95), loc=2, borderaxespad=0, prop={'size':12}, frameon=False)
L.get_texts()[0].set_text('Stereo-normal')
L.get_texts()[1].set_text('Stereo-anomalous')
L.get_texts()[2].set_text('')
L.get_texts()[3].set_text('')

#plt.show()
plt.savefig(fname=results_dir + 'd_prime_sa_level_PPR_2.png', bbox_inches='tight', format='png', dpi=300)



