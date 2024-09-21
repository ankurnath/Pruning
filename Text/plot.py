from utils import *

import seaborn as sns
import matplotlib.pyplot as plt

dataset = 'food101'
df = load_from_pickle(f'image_{dataset}')

# df['Budget'].append(budget)
# df['Ratio_Obj(QS)'].append(obj_val_FS_QS/obj_val_FS)
# df['Pg(QS)'].append(size_QS)
# df['Ratio_Obj(SS)'].append(obj_val_FS_SS/obj_val_FS)
# df['Pg(SS)'].append(size_SS)
# df['Obj'].append(obj_val_FS)
# df['Ratio_Obj(Random)'].append(obj_val_FS_Random/obj_val_FS)

plt.figure(dpi=300)

# Plotting the ratios with larger markers and improved styling
sns.lineplot(data=df, x='Budget', y='Ratio_Obj(QS)', label=f'QS$(P_g={df.iloc[-1]["Pg(QS)"]}\%)$', linestyle='--', marker='o', 
             markersize=8, color='#f46666')
sns.lineplot(data=df, x='Budget', y='Ratio_Obj(SS)', label=f'SS$(P_g={df.iloc[-1]["Pg(SS)"]}\\%)$',linestyle='--' ,marker='s', 
             markersize=8, color='#37a1e2')
sns.lineplot(data=df, x='Budget', y='Ratio_Obj(Random)', label=f'Random$(P_g={df.iloc[-1]["Pg(QS)"]}\%)$',linestyle='--', marker='D', 
             markersize=8, color='#f67f10')

# Labeling the plot
# plt.title('Ratios of Different Objectives Relative to Obj(FS)')
# plt.xlabel('Budget')
# plt.ylabel('Ratio')
# plt.legend(title='Objective', title_fontsize='13', fontsize='11')

fontsize = 16

# plt.title('Multi Budget results for Image retrival system \n $P_g =$ percentage of the ground set')
plt.title(f'Image Retrieval System Dataset:{dataset} \n($P_g$: Pruned Ground Set in percentage)')

plt.xlabel('Budget',fontsize=fontsize)
plt.ylabel('Approximation ratio',fontsize=fontsize)

plt.xticks(fontsize=fontsize )
plt.yticks(fontsize=fontsize )

# Adding a light grid
plt.grid(True, linestyle='--', alpha=0.7)
plt.locator_params(nbins=6)
# Legend
plt.legend(fontsize='14')
sns.despine()

# Show the plot
plt.savefig(f'{dataset}')
plt.show()

