import pandas as pd
import os
from collections import defaultdict
import pickle

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



for problem in [
               # 'MaxCover',
               # 'MaxCut',
               'IM'
               ]:
    datasets = ['Facebook','Wiki','Deezer','Slashdot','Twitter','DBLP','YouTube',
                # 'Skitter'
                ]
    df = defaultdict(list)
    for dataset in datasets:
        
        file_path = os.path.join(problem,'data',dataset,'knapsack_multi','Quickfilter_aistats')
        try:
            df_ = load_from_pickle(file_path=file_path)
        except:
           continue

        df['Dataset'].append(df_['Dataset'].iloc[-1])
        df['P_r Multi'].append(df_['Ratio Multi'].iloc[-1])
        df['P_g Multi(%)'].append(100-df_['Pruned Ground set Multi(%)'].iloc[-1])
        df['P_r Multi(TOP-K)'].append(df_['Ratio Multi(TOP-K)'].iloc[-1])
        
        # df['P_r Single'].append(df_['Ratio Single'].iloc[-1])
      #   df['P_r(TOP-K)'].append(df_['Ratio Single(TOP-K)'].iloc[-1])
    

    df = pd.DataFrame(df)

    print(df)
    break


        
