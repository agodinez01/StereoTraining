import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

main_dir = "C:/Users/angie/Git Root/StereoTraining/data/"
results_dir = "C:/Users/angie/Git Root/StereoTraining/figs/"

# Load variables
os.chdir('C:/Users/angie/Git Root/StereoTraining/data')
dData = pd.read_csv('d_prime_santiago.csv')

level_id = dData.level.unique()
sa_id = dData.sa.unique()
group_id = dData.group.unique()

def makeDf():
    groupVals = []
    levelVals = []
    saVals = []
    d_primePost = []
    d_primePre = []
    d_primePPR = []
    d_primePre_sd = []
    d_primePost_sd = []

    for g in group_id:
        for l in level_id:
            for s in sa_id:

                primePre = dData['d prime pre'][(dData.group == g) & (dData.level == l) & (dData.sa == s)].mean()
                primePost = dData['d prime post'][(dData.group == g) & (dData.level == l) & (dData.sa == s)].mean()
                primePreSD = dData['d prime pre'][(dData.group == g) & (dData.level == l) & (dData.sa == s)].std()
                primePostSD = dData['d prime post'][(dData.group == g) & (dData.level == l) & (dData.sa == s)].std()

                groupVals.append(g)
                levelVals.append(l)
                saVals.append(s)
                d_primePre.append(primePre)
                d_primePost.append(primePost)
                d_primePre_sd.append(primePreSD)
                d_primePost_sd.append(primePostSD)

    return groupVals, levelVals, saVals, d_primePre, d_primePost, d_primePre_sd, d_primePost_sd

group_list, level_list, sa_list, dPre, dPost, dPreSD, dPostSD = makeDf()
d = {'group': group_list, 'level': level_list, 'sa': sa_list, 'dPre': dPre, 'dPost':dPost, 'dPreSD': dPreSD, 'dPostSD':dPostSD}
df = pd.DataFrame(d)

PPR = df.dPost/df.dPre
df['PPR'] = PPR

df.to_csv(main_dir + 'd_prime_PPR_by_level_and_demand.csv', index=False)

sns.boxplot(x='sa', y='dPre', hue='group', data=df)

plt.xlabel('stereo demand (arc secs)')
plt.ylabel('d\' prime pre')
plt.ylim([1.2, 3.2])

L = plt.legend(bbox_to_anchor=(0.05, 0.95), loc=2, borderaxespad=0, prop={'size':10}, frameon=False)
L.get_texts()[0].set_text('Stereo-normal')
L.get_texts()[1].set_text('stereo-anomalous')

#plt.show()
plt.savefig(fname=results_dir + 'd_prime_pre.png', bbox_inches='tight', format='png', dpi=300)
plt.clf()

sns.boxplot(x='sa', y='dPost', hue='group', data=df)
plt.xlabel('stereo demand (arc secs)')
plt.ylabel('d\' prime post')
plt.ylim([1.2, 3.2])

L = plt.legend(bbox_to_anchor=(0.05, 0.95), loc=2, borderaxespad=0, prop={'size':10}, frameon=False)
L.get_texts()[0].set_text('Stereo-normal')
L.get_texts()[1].set_text('stereo-anomalous')

plt.savefig(fname=results_dir + 'd_prime_post.png', bbox_inches='tight', format='png', dpi=300)

plt.clf()

sns.boxplot(x='sa', y='PPR', data=df, hue='group')
plt.axhline(y=1, color='k', linestyle='--')
plt.xlabel('stereo demand (arc secs)')
plt.ylabel('Post:pre ratio')

L = plt.legend(bbox_to_anchor=(0.05, 0.95), loc=2, borderaxespad=0, prop={'size':10}, frameon=False)
L.get_texts()[0].set_text('Stereo-normal')
L.get_texts()[1].set_text('stereo-anomalous')

plt.savefig(fname=results_dir + 'd_prime_ppr.png', bbox_inches='tight', format='png', dpi=300)



