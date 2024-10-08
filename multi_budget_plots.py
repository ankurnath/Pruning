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
    # print(f'Data has been loaded from {file_path}')
    return loaded_data
import os
from collections import defaultdict
for problem in [
                'MaxCover',
                'MaxCut',
                'IM'
              ]:
    image_folder = f'{problem}/figures'
    os.makedirs(image_folder,exist_ok=True)
    root_folder=os.path.join(problem,'data')
    # datasets=['Facebook','DBLP','Skitter','YouTube']
    for folder in [
                    # 'knapsack',
                    'knapsack_multi'
                    ]:
        datasets=os.listdir(root_folder)

        
        # df ={'algorithm':[],'Size of Ground Set':[],'Ratio':[],'Queries':[]}
        df =defaultdict(list)

        for dataset in datasets:

            # print('*'*20)
            # print(dataset)
            dataset_path = os.path.join(root_folder,dataset)
            print(dataset_path)
            try:
                # print(os.path.join(dataset_path,folder,'Quickfilter_degree'))
                # df_ = load_from_pickle(os.path.join(dataset_path,folder,'GNNpruner'))
                df_ = load_from_pickle(os.path.join(dataset_path,folder,'Quickfilter_aistats'))
                
                df_['Budget'] = [10,30,50,70,90,100]
                # print(,folder,df_['Dataset'].iloc[0])
                # df['dataset'].append(dataset)
                # # df['algorithm'].append('QS')
                # # df['Delta'].append(df_['Delta'].iloc[0])
                # df['Size of Ground Set'].append(df_['Pruned Ground set(%)'].iloc[0])
                
                # df['Ratio'].append(df_['Ratio(%)'].iloc[0])
                # # df['Queries'].append(df_['Queries(%)'].iloc[0])
                plt.figure(dpi=200, figsize=(8,6))
                df_['Pruned Ground set Multi(%)'] = 100 - df_['Pruned Ground set Multi(%)']
                df_['Pruned Ground set Single(%)'] = 100 - df_['Pruned Ground set Single(%)']

                # Scatter for Multi-Budget
                plt.scatter(df_['Budget'], df_['Ratio Multi'], marker='o', s=500, color='blue', alpha=0.7)
                plt.plot(df_['Budget'], df_['Ratio Multi'], linestyle='--', color='blue', 
                        label=f"Multi $(P_g={df_['Pruned Ground set Multi(%)'].iloc[0]:.2f}\%)$")

                # Scatter for Single-Budget
                plt.scatter(df_['Budget'], df_['Ratio Single'], marker='*', s=600, color='red', alpha=0.7)
                plt.plot(df_['Budget'], df_['Ratio Single'], linestyle='--',color='red', 
                        label=f"Single $(P_g ={df_['Pruned Ground set Single(%)'].iloc[0]:.2f}\%)$")


                fontsize = 30
                # Adding grid, legend, and style
                plt.grid(alpha=0.7, linestyle='--')
                sns.despine()
                plt.legend(frameon = False,fontsize = fontsize)
                plt.title(df_['Dataset'].iloc[0],fontsize = fontsize+4)
                plt.xlabel('Budgets', fontsize=fontsize )
                plt.ylabel('Value Frac., $P_r$ (%)', fontsize=fontsize)
                plt.xticks(fontsize=fontsize )
                plt.yticks(fontsize=fontsize )
                plt.locator_params(nbins=6)
                
                file_name = os.path.join(image_folder,df_['Dataset'].iloc[0])
                
                # plt.savefig(f'{file_name}', bbox_inches='tight')
                plt.savefig(f'{file_name}.pdf', bbox_inches='tight',dpi=300)
                plt.savefig(f'{file_name}.png', bbox_inches='tight',dpi=300)

                plt.close()
            except:
                pass
                
        # # print(df)
        # df = pd.DataFrame(df)
        # print(df)
        # # print('-'*20)
