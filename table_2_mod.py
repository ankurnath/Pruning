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
def f(ratio_value,pruned_value):
        
        # value = np.log(val1/(100-val2))
        print(f'& {ratio_value:.4f}' ,end ='')
        print(f'& {pruned_value:.4f}' ,end ='')

        value = ratio_value*pruned_value

        print('& \multicolumn{1}{c|}')
        print('{',end='')
        print(f'{value:.4f}',end='')
        print('}',end='')


for problem in [
                'MaxCover',
                'MaxCut',
                'IM'
               ]:
     
    print('& \multicolumn{9}{c|}',end='')
    print('{\\textbf{',end='')
    print(f'{problem}',end='')
    print('}} ',end=' \\\\ \hline')
    print()

    datasets = ['Facebook','Wiki','Deezer','Slashdot','Twitter','DBLP','YouTube','Skitter']








    for dataset in datasets:

        print(' \multicolumn{1}{|c|}',end='')
        print('{',end='')
        print(f'{dataset}',end='')
        print('} ',end='')
        print()

      

        df_ = load_from_pickle(os.path.join(problem,'data',dataset,
                                            'knapsack_multi','Quickfilter_aistats'))
        # print(df_.columns)
        
        qs_ratio = df_["Ratio Multi"].iloc[-1]/100
        qs_pruned = (100-df_["Pruned Ground set Multi(%)"].iloc[-1])/100
        f(qs_ratio,qs_pruned)

        topk_ratio = df_["Ratio Multi(TOP-K)"].iloc[-1]/100
        df_ = load_from_pickle(os.path.join(problem,'data',dataset,
                                            'Knapsack_GNN','GNNpruner_aistats'))
        f(topk_ratio,qs_pruned)

        gnn_ratio = df_["Ratio(%)"].iloc[-1]/100
        gnn_pruned = (100-df_["Pruned Ground set(%)"].iloc[-1])/100

        f(gnn_ratio ,gnn_pruned)
        
        # print(f'& {df_["Ratio Multi"].iloc[-1]:.2f}' ,end = '')
        # print(f'& {(100-df_["Pruned Ground set Multi(%)"].iloc[-1]):.2f}' ,end = '')

        # f(df_["Ratio Multi"].iloc[-1], df_["Pruned Ground set Multi(%)"].iloc[-1])

        # print(f'& {df_["Ratio Multi(TOP-K)"].iloc[-1]:.2f}' ,end = '')
        # print(f'& {(100-df_["Pruned Ground set Multi(%)"].iloc[-1]):.2f}' ,end = '')

        # f(df_["Ratio Multi(TOP-K)"].iloc[-1], df_["Pruned Ground set Multi(%)"].iloc[-1])

        

        # print(f'& {df_["Ratio(%)"].iloc[-1]:.2f}' ,end = '')

        # print(f'& {(100-df_["Pruned Ground set(%)"].iloc[-1]):.2f}' ,end = '')
        # f(df_["Ratio(%)"].iloc[-1], df_["Pruned Ground set(%)"].iloc[-1])


        print(' \\\\') 


    print('\hline')

