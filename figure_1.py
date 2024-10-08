
import os
from collections  import defaultdict
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
def load_from_pickle(file_path):
    """
    Load data from a pickle file.

    Parameters:
    - file_path: The path to the pickle file.

    Returns:
    - loaded_data: The loaded data.
    """
    with open(file_path, 'rb') as file:
        loaded_data = pickle.load(file)
    print(f'Data has been loaded from {file_path}')
    return loaded_data
gaint_df = []


for problem in [
                'MaxCover', 
                # 'MaxCut',
                # 'IM'
              ]:
    root_folder=os.path.join(problem,'data')
    # datasets=['Facebook','DBLP','Skitter','YouTube']
    
    datasets=os.listdir(root_folder)
    # print(datasets)
    # datasets = ['YouTube']

    for dataset in datasets:
        # if dataset in ['Slashdot']:
        

        

        print('*'*20)
        print(dataset)
        dataset_path = os.path.join(root_folder,dataset)
        algorthims = os.listdir(dataset_path)

        # df ={'algorithm':[],'Size of Ground Set':[],'Ratio':[],'Queries':[]}
        df = defaultdict(list)
        # for algorthim in ['Quickfilter','SS','LeNSE','CombHelperTeacher','CombHelperStudent','GNNpruner']:
        for algorthim in ['Quickfilter','SS','CombHelperStudent','GNNpruner']:
          try:
            df_ = load_from_pickle(os.path.join(dataset_path,algorthim))
          except:
            print(f'Failed to load from {os.path.join(dataset_path,algorthim)}')
            continue

          # columns =['Ground set(Pruned)','Ratio(%)','Queries(%)']
          df['dataset'].append(dataset)
          if algorthim == 'CombHelperStudent':
              df['algorithm'].append('COMBHelper')

          elif algorthim =='Quickfilter':
              df['algorithm'].append('QuickPrune(OURS)')

          elif algorthim =='GNNpruner':
              df['algorithm'].append('GNNPruner')
              
          else:
            df['algorithm'].append(algorthim)
          
          df['Ratio'].append(df_['Ratio(%)'].iloc[0])
          df['Size of Ground Set'].append(df_['Pruned Ground set(%)'].iloc[0])
          df['TimeToPrune'].append(df_['TimeToPrune'].iloc[0])
            # df['Multibudget'].append(df_['Multibudget'].iloc[0])
            
            
            # df['Objective Value(Unpruned)'].append(df_['Objective Value(Unpruned)'].iloc[0])
            # df['Objective Value(Pruned)'].append(df_['Objective Value(Pruned)'].iloc[0])
            # df['Queries'].append(df_['Queries(%)'].iloc[0].zfill(4))
            # df['Queries'].append(df_['Queries(%)'].iloc[0])
          
          # else:
          #    pass
      # print(df)
        df = pd.DataFrame(df)
        # print(df)
        # df['Queries'] = df['Queries'].apply(lambda x: f"{x:.4f}")
        df['Size of Ground Set']=df['Size of Ground Set']
        # df['Queries'] = df['Queries'].round(4)
        # print(df)
        # print('-'*20)
        gaint_df.append(df)


gcomb = { 'MaxCover': [['Facebook', 92.7, 93],
                       ['Wiki', 99, 97],
                       ['Deezer', 99.4, 87],
                       ['Slashdot', 100, 98],
                       ['Twitter', 99.7, 83],
                       ['DBLP', 99.9, 93.0],
                       ['YouTube', 99.8, 93],
                       ['Skitter', 99.9, 90.00]],

          'MaxCut':[['Facebook', 81.30, 5],
                    ['Wiki', 92, 4],
                    ['Deezer', 85, 1],
                    ['Slashdot', 63.20, 1],
                    ['Twitter', 62.80, 1],
                    ['DBLP', 64.60, 1],
                    ['YouTube', 53.6, 1],
                    ['Skitter', 42.70, 1]],

          'IM':[['Facebook', 95.1, 27],
                ['Wiki', 96.9, 10],
                ['Deezer', 80.5, 5.00],
                ['Slashdot', 96.6, 2.00],
                ['Twitter', 92.0, 2.00],
                ['DBLP', 86.3, 1],
                ['YouTube', 93.3, 1],
                ['Skitter', 88.3, 1]],
}

lense = { 'MaxCover': [['Facebook', 96.6, 93.00],
                       ['Wiki', 109.4, 66],
                       ['Deezer', 97.9, 25],
                       ['Slashdot', 97.9, 31],
                       ['Twitter', 98.9, 67],
                       ['DBLP', 99.0, 10],
                       ['YouTube', 98.2, 21],
                       ['Skitter', 97.6, 30.00]],

          'MaxCut':[['Facebook', 100, 84.62],
                    ['Wiki', 98.10, 61],
                    ['Deezer', 97.5, 26],
                    ['Slashdot', 99.00, 38],
                    ['Twitter', 98.70, 52],
                    ['DBLP', 99.3, 8],
                    ['YouTube', 98.7, 21.00],
                    ['Skitter', 97.4, 29.00]],

          'IM':[['Facebook', 97.9, 91],
                ['Wiki', 96.0, 49],
                ['Deezer', 97.2, 24],
                ['Slashdot', 96.6, 23],
                ['Twitter', 96.6, 60.00],
                ['DBLP', 96.9, 21],
                ['YouTube', 97.1, 25],
                ['Skitter', 98.3, 22]],
}


df = defaultdict(list)
for dataset,ratio,ground_set in gcomb[problem]:
    df['algorithm'].append('GCOMB-P')
    df['dataset'].append(dataset)
    df['Ratio'].append(ratio)
    df['Size of Ground Set'].append(ground_set)

df = pd.DataFrame(df)
gaint_df.append(df)
df = defaultdict(list)
for dataset,ratio,ground_set in lense[problem]:
    df['algorithm'].append('LeNSE')
    df['dataset'].append(dataset)
    df['Ratio'].append(ratio)
    df['Size of Ground Set'].append(ground_set)
df = pd.DataFrame(df)
gaint_df.append(df) 

gaint_df = pd.concat(gaint_df)

# Sample dataset (replace this with your actual DataFrame)
df = gaint_df.copy()
df = df[df['dataset']=='YouTube']

# Define the specific order for datasets
dataset_order = ['YouTube']

# Convert the dataset column to a categorical type with the specified order
df['RatioFrac'] = df['Ratio']/100

print(df['RatioFrac'])
df['Size of Ground SetFrac'] = 1 - df['Size of Ground Set']/100
df['dataset'] = pd.Categorical(df['dataset'], categories=dataset_order, ordered=True)
print(df)
# Prepare the data for plotting by melting it
df_melted = pd.melt(df, id_vars=['algorithm'], value_vars=['RatioFrac', 'Size of Ground SetFrac'], 
                    var_name='Metric', value_name='Value')

# Create the plot
plt.figure(dpi=300)
plt.ylim(0.6,1)
bar_plot = sns.barplot(x='algorithm', y='Value', 
                       hue='Metric', data=df_melted, 
                       palette=['#f79903','#f9766e'],
                       errorbar=None,
                       edgecolor='black'
                       )




fontsize =20 

plt.ylabel('Ratio',fontsize=20)
plt.xlabel('')
plt.xticks()
plt.yticks(fontsize =18)
# plt.yscale('log')
sns.despine()

plt.legend(['$P_r=f(\mathcal{H}(\mathcal{U}_{pruned}))/f(\mathcal{H}(\mathcal{U}))$','$P_g=|\mathcal{U}-\mathcal{U}_{pruned}|/|\mathcal{U}|$'],loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1,fontsize=fontsize,frameon=False)
# plt.legend(['$P_r$','$P_g$'],frameon=False)

# plt.axhline(y=df[df['dataset']=='YouTube']['RatioFrac'].iloc[0], color='red', linestyle='--')
# plt.axhline(y=df[df['dataset']=='YouTube']['Size of Ground SetFrac'].iloc[0], color='blue', linestyle='dotted')

# plt.set
plt.locator_params(axis='y', nbins=4)
plt.tight_layout()
plt.savefig(f'{problem}_performance.pdf',bbox_inches='tight', dpi=300)
# plt.savefig(f'{problem}_performance.png',bbox_inches='tight', dpi=300)
# Show the plot
plt.show()


