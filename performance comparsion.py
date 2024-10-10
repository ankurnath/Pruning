import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create the dataframe
data = {
    "Algorithm": ["QuickPrune", "SS", "GCOMB-P", "LeNSE", "COMBHelper", "GNNPruner"],
    "CPU": [3.1, 3.2, 3.12, 4.1, 6.3, 5.6],
    "GPU": [0, 0, 0, 1122, 3658, 1704]
}

df = pd.DataFrame(data)

# Multiply the CPU values by 64 * 1024
df['CPU'] = df['CPU'] * (64 * 1024)/100

# Melt the dataframe to make it suitable for plotting
df_melted = df.melt(id_vars="Algorithm", value_vars=["CPU", "GPU"], var_name="Resource", value_name="Usage")

# Create the plot
fontsize = 18
plt.figure(dpi=300)
sns.barplot(x="Algorithm", y="Usage", hue="Resource", data=df_melted,edgecolor='black')
# plt.title('CPU and GPU Usage by Algorithm')
plt.ylabel('Average Usage(MB)',fontsize=fontsize)
plt.xlabel('')
plt.xticks(rotation=90,fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.tight_layout()
sns.despine()
# plt.legend(['CPU','GPU'],fontsize=fontsize)
# Save the plot as an image
plt.grid(alpha=0.3, linestyle='--',axis='y')
plt.locator_params(nbins=6)
plt.savefig('cpu_gpu_usage_by_algorithm.pdf', bbox_inches='tight',dpi=300)

# Show the plot
# plt.show()
