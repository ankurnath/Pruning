import os
from collections  import defaultdict
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
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



datasets = ['Facebook','Wiki','Deezer','Slashdot','Twitter','DBLP','YouTube',
            # 'Skitter'
            ]
def f(val1,val2):
        
        # value = np.log(val1/(100-val2))
        value = np.log(val1*(100-val2))

        print('& \multicolumn{1}{c|}')
        print('{',end='')
        print(f'{value:.2f}',end='')
        print('}',end='')

for dataset in datasets:

    print(' \multicolumn{1}{l|}',end='')
    print('{',end='')
    print(f'{dataset}',end='')
    print('} ',end='')
    print()

    for problem in [
                    'MaxCover',
                    'MaxCut',
                    'IM']:

        df_ = load_from_pickle(os.path.join(problem,'data',dataset,
                                            'knapsack_multi','Quickfilter_aistats'))
        
        print(f'& {df_["Ratio Multi"].iloc[-1]:.2f}' ,end = '')
        print(f'& {(100-df_["Pruned Ground set Multi(%)"].iloc[-1]):.2f}' ,end = '')

        f(df_["Ratio Multi"].iloc[-1], df_["Pruned Ground set Multi(%)"].iloc[-1])

        print(f'& {df_["Ratio Multi(TOP-K)"].iloc[-1]:.2f}' ,end = '')
        print(f'& {(100-df_["Pruned Ground set Multi(%)"].iloc[-1]):.2f}' ,end = '')

        f(df_["Ratio Multi(TOP-K)"].iloc[-1], df_["Pruned Ground set Multi(%)"].iloc[-1])

        df_ = load_from_pickle(os.path.join(problem,'data',dataset,
                                            'Knapsack_GNN','GNNpruner_aistats'))

        print(f'& {df_["Ratio(%)"].iloc[-1]:.2f}' ,end = '')

        print(f'& {(100-df_["Pruned Ground set(%)"].iloc[-1]):.2f}' ,end = '')
        f(df_["Ratio(%)"].iloc[-1], df_["Pruned Ground set(%)"].iloc[-1])


    print(' \\\\') 




