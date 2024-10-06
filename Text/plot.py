from utils import *

import seaborn as sns
import matplotlib.pyplot as plt

# dataset = 'food101'


# dataset ='beans'

dataset = 'cifar100'
# dataset = 'UCF101'

try:
    df = load_from_pickle(f'image_{dataset}')
except:
    df = load_from_pickle(f'video_{dataset}')


# df = load_from_pickle(f'video_{dataset}')


plt.figure(dpi=300)

markersize = 25
# Plotting the ratios with larger markers and improved styling
sns.lineplot(data=df, x='Budget', y='Ratio_Obj(QS)', label=f'QS$(P_g={df.iloc[-1]["Pg(QS)"]:.2f}\%)$', linestyle='--', marker='o', 
             markersize=markersize, color='#f46666')

try:
    sns.lineplot(data=df, x='Budget', y='Ratio_Obj(SS)', label=f'SS$(P_g={df.iloc[-1]["Pg(SS)"]:.2f}\\%)$',linestyle='--' ,marker='^', 
             markersize=markersize, color='#37a1e2')
except:
    pass

try:
    sns.lineplot(data=df, x='Budget', y='Ratio_Obj(top-k)', label=f'TOP-K$(P_g={df.iloc[-1]["Pg(QS)"]:.2f}\\%)$',linestyle='--' ,marker='^', 
             markersize=markersize, color='#4d6b23')
except:
    pass
sns.lineplot(data=df, x='Budget', y='Ratio_Obj(Random)', label=f'Random$(P_g={df.iloc[-1]["Pg(QS)"]:.2f}\%)$',linestyle='--', marker='D', 
             markersize=markersize, color='#f67f10')

# Labeling the plot
# plt.title('Ratios of Different Objectives Relative to Obj(FS)')
# plt.xlabel('Budget')
# plt.ylabel('Ratio')
# plt.legend(title='Objective', title_fontsize='13', fontsize='11')

fontsize = 30

# plt.title('Multi Budget results for Image retrival system \n $P_g =$ percentage of the ground set')
# plt.title(f'Image Retrieval System Dataset:{dataset} \n($P_g$: Pruned Ground Set in percentage)')

plt.title(f'{dataset.upper()}',fontsize=fontsize)

plt.xlabel('Budget',fontsize=fontsize)
plt.ylabel('Approx. Ratio,$P_r$',fontsize=fontsize)

plt.xticks(fontsize=fontsize )
plt.yticks(fontsize=fontsize )

# Adding a light grid
plt.grid(alpha=0.3, linestyle='--')
plt.locator_params(nbins=6)
# Legend
# plt.legend(fontsize=fontsize-4,frameon= False,location='bottom left')
plt.legend(fontsize=fontsize-4, frameon=False, loc='center left')

sns.despine()

# Show the plot
plt.savefig(f'{dataset.upper()}.pdf', bbox_inches='tight',dpi=300)
# plt.show()

