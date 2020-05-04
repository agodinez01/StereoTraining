import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

main_dir = "C:/Users/angie/Git Root/StereoTraining/data"
results_dir = "C:/Users/angie/Git Root/StereoTraining/figs/"

# Load variables
os.chdir('C:/Users/angie/Git Root/StereoTraining/data')
dData = pd.read_csv('d_prime_santiago.csv')

# Set variables
levels = dData.level.unique()
sas = dData.sa.unique()
subjects = dData.subject.unique()
id = dData.id.unique()
choices = dData.choices.unique()

plotMarkers = ['o', 's', 'D']

a = dData[dData.id == 'N7']
aa = a[a.sa > 300]

frames = [aa, dData.loc[dData.id == 'AMS1'], dData.loc[dData.id == 'AA4'], dData.loc[dData.id == 'ASW1']]
famous4 = pd.concat(frames)

# g = sns.FacetGrid(famous4, col='id', hue='level')
# g =g.map(plt.scatter, 'sa', 'd prime pre')
# g =g.map(plt.scatter, 'sa', 'd prime post')
# plt.show()

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex= True, sharey= True)
ax1.plot(famous4.sa[(famous4.id == 'N7') & (famous4.level == 1)], famous4['d prime pre'][(famous4.id == 'N7') & (famous4.level == 1)], plotMarkers[0], c='#4E504E', alpha=0.75)
ax1.plot(famous4.sa[(famous4.id == 'N7') & (famous4.level == 2)], famous4['d prime pre'][(famous4.id == 'N7') & (famous4.level == 2)], plotMarkers[1], c='#4E504E', alpha=0.75)
ax1.plot(famous4.sa[(famous4.id == 'N7') & (famous4.level == 3)], famous4['d prime pre'][(famous4.id == 'N7') & (famous4.level == 3)], plotMarkers[2], c='#4E504E', alpha=0.75)

ax1.plot(famous4.sa[(famous4.id == 'N7') & (famous4.level == 1)] + 30, famous4['d prime post'][(famous4.id == 'N7') & (famous4.level == 1)], plotMarkers[0], c='#4E504E', markerfacecolor='None', alpha=0.75)
ax1.plot(famous4.sa[(famous4.id == 'N7') & (famous4.level == 2)] + 30, famous4['d prime post'][(famous4.id == 'N7') & (famous4.level == 2)], plotMarkers[1], c='#4E504E', markerfacecolor='None', alpha=0.75)
ax1.plot(famous4.sa[(famous4.id == 'N7') & (famous4.level == 3)] + 30, famous4['d prime post'][(famous4.id == 'N7') & (famous4.level == 3)], plotMarkers[2], c='#4E504E', markerfacecolor='None', alpha=0.75)


# ax1.plot(famous4.sa[(famous4.id == 'N7') & (famous4.level == i)], famous4['d prime pre'][(famous4.id == 'N7') & (famous4.level == i)], 'o', c='#4E504E', alpha=0.75)
#ax1.plot(famous4.sa[famous4.id == 'N7'] + 30, famous4['d prime post'][famous4.id == 'N7'], 'o', c='#4E504E', markerfacecolor='None')
ax2.plot(famous4.sa[(famous4.id == 'ASW1') & (famous4.level == 1)], famous4['d prime pre'][(famous4.id == 'ASW1') & (famous4.level == 1)], plotMarkers[0], c='#0B940B', alpha=0.75)
ax2.plot(famous4.sa[(famous4.id == 'ASW1') & (famous4.level == 2)], famous4['d prime pre'][(famous4.id == 'ASW1') & (famous4.level == 2)], plotMarkers[1], c='#0B940B', alpha=0.75)
ax2.plot(famous4.sa[(famous4.id == 'ASW1') & (famous4.level == 3)], famous4['d prime pre'][(famous4.id == 'ASW1') & (famous4.level == 3)], plotMarkers[2], c='#0B940B', alpha=0.75)

ax2.plot(famous4.sa[(famous4.id == 'ASW1') & (famous4.level == 1)] + 30, famous4['d prime post'][(famous4.id == 'ASW1') & (famous4.level == 1)], plotMarkers[0], c='#0B940B', markerfacecolor='None', alpha=0.75)
ax2.plot(famous4.sa[(famous4.id == 'ASW1') & (famous4.level == 2)] + 30, famous4['d prime post'][(famous4.id == 'ASW1') & (famous4.level == 2)], plotMarkers[1], c='#0B940B', markerfacecolor='None', alpha=0.75)
ax2.plot(famous4.sa[(famous4.id == 'ASW1') & (famous4.level == 3)] + 30, famous4['d prime post'][(famous4.id == 'ASW1') & (famous4.level == 3)], plotMarkers[2], c='#0B940B', markerfacecolor='None', alpha=0.75)



#ax2.plot(famous4.sa[famous4.id == 'ASW1'], famous4['d prime pre'][famous4.id == 'ASW1'], 'o', c='#0B940B', alpha=0.75)
#ax2.plot(famous4.sa[famous4.id == 'ASW1'] + 30, famous4['d prime post'][famous4.id == 'ASW1'], 'o', c = '#0B940B', markerfacecolor='None')

ax3.plot(famous4.sa[(famous4.id == 'AA4') & (famous4.level == 1)], famous4['d prime pre'][(famous4.id == 'AA4') & (famous4.level == 1)], plotMarkers[0], c='#0F1BD8', alpha=0.75)
ax3.plot(famous4.sa[(famous4.id == 'AA4') & (famous4.level == 2)], famous4['d prime pre'][(famous4.id == 'AA4') & (famous4.level == 2)], plotMarkers[1], c='#0F1BD8', alpha=0.75)
ax3.plot(famous4.sa[(famous4.id == 'AA4') & (famous4.level == 3)], famous4['d prime pre'][(famous4.id == 'AA4') & (famous4.level == 3)], plotMarkers[2], c='#0F1BD8', alpha=0.75)

ax3.plot(famous4.sa[(famous4.id == 'AA4') & (famous4.level == 1)] + 30, famous4['d prime post'][(famous4.id == 'AA4') & (famous4.level == 1)], plotMarkers[0], c='#0F1BD8', markerfacecolor='None', alpha=0.75)
ax3.plot(famous4.sa[(famous4.id == 'AA4') & (famous4.level == 2)] + 30, famous4['d prime post'][(famous4.id == 'AA4') & (famous4.level == 2)], plotMarkers[1], c='#0F1BD8', markerfacecolor='None', alpha=0.75)
ax3.plot(famous4.sa[(famous4.id == 'AA4') & (famous4.level == 3)] + 30, famous4['d prime post'][(famous4.id == 'AA4') & (famous4.level == 3)], plotMarkers[2], c='#0F1BD8', markerfacecolor='None', alpha=0.75)

# #ax3.plot([0, 4], [0, 4], 'k--')
# ax3.plot(famous4.sa[famous4.id == 'AA4'], famous4['d prime pre'][famous4.id == 'AA4'], 'o', c='#0F1BD8', alpha=0.75)
# ax3.plot(famous4.sa[famous4.id == 'AA4'] + 30, famous4['d prime post'][famous4.id == 'AA4'], 'o', c = '#0F1BD8', markerfacecolor='None')

ax4.plot(famous4.sa[(famous4.id == 'AMS1') & (famous4.level == 1)], famous4['d prime pre'][(famous4.id == 'AMS1') & (famous4.level == 1)], plotMarkers[0], c='#D20000', alpha=0.75)
ax4.plot(famous4.sa[(famous4.id == 'AMS1') & (famous4.level == 2)], famous4['d prime pre'][(famous4.id == 'AMS1') & (famous4.level == 2)], plotMarkers[1], c='#D20000', alpha=0.75)
ax4.plot(famous4.sa[(famous4.id == 'AMS1') & (famous4.level == 3)], famous4['d prime pre'][(famous4.id == 'AMS1') & (famous4.level == 3)], plotMarkers[2], c='#D20000', alpha=0.75)

ax4.plot(famous4.sa[(famous4.id == 'AMS1') & (famous4.level == 1)] + 30, famous4['d prime post'][(famous4.id == 'AMS1') & (famous4.level == 1)], plotMarkers[0], c='#D20000', markerfacecolor='None', alpha=0.75)
ax4.plot(famous4.sa[(famous4.id == 'AMS1') & (famous4.level == 2)] + 30, famous4['d prime post'][(famous4.id == 'AMS1') & (famous4.level == 2)], plotMarkers[1], c='#D20000', markerfacecolor='None', alpha=0.75)
ax4.plot(famous4.sa[(famous4.id == 'AMS1') & (famous4.level == 3)] + 30, famous4['d prime post'][(famous4.id == 'AMS1') & (famous4.level == 3)], plotMarkers[2], c='#D20000', markerfacecolor='None', alpha=0.75)

# ax4.plot(famous4.sa[famous4.id == 'AMS1'], famous4['d prime pre'][famous4.id == 'AMS1'], 'o', c='#D20000', alpha=0.75)
# ax4.plot(famous4.sa[famous4.id == 'AMS1'] + 30, famous4['d prime post'][famous4.id == 'AMS1'], 'o', c = '#D20000', markerfacecolor='None')


for ax in fig.get_axes():
    ax.label_outer()
    ax.set_xticks([1000, 800, 600, 400])
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_xlabel('Disparity (arc secs)')
    ax.set_ylabel('d prime')

plt.legend()
L = plt.legend(bbox_to_anchor=(1.01, 2.25), loc=2, borderaxespad=0, prop={'size': 10}, frameon=False)
L.get_texts()[0].set_text('Pre: Level 1')
L.get_texts()[1].set_text('Pre: Level 2')
L.get_texts()[2].set_text('Pre: Level 3')
L.get_texts()[3].set_text('Post: Level 1')
L.get_texts()[4].set_text('Post: Level 2')
L.get_texts()[5].set_text('Post: Level 3')

#plt.show()
plt.savefig(fname=results_dir + 'famous4_dprime.png', bbox_inches='tight', format='png', dpi=300)

