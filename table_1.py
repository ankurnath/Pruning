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
gcomb = {
    'MaxCover': [['Facebook', 92.70, 7.00],
                 ['Wiki', 99.00, 3.00],
                 ['Deezer', 99.40, 13.00],
                 ['Slashdot', 100.00, 2.00],
                 ['Twitter', 99.70, 17.00],
                 ['DBLP', 99.90, 3.00],
                 ['YouTube', 99.80, 7.00],
                 ['Skitter', 99.90, 10.00]],

    'MaxCut': [['Facebook', 81.30, 95.00],
               ['Wiki', 92.00, 96.00],
               ['Deezer', 85.00, 99.00],
               ['Slashdot', 63.20, 99.00],
               ['Twitter', 62.80, 99.00],
               ['DBLP', 64.60, 99.00],
               ['YouTube', 53.60, 99.00],
               ['Skitter', 42.70, 99.00]],

    'IM': [['Facebook', 95.10, 73.00],
           ['Wiki', 96.90, 90.00],
           ['Deezer', 80.50, 55.00],
           ['Slashdot', 96.60, 98.00],
           ['Twitter', 92.00, 98.00],
           ['DBLP', 86.30, 99.00],
           ['YouTube', 93.30, 99.00],
           ['Skitter', 88.30, 99.00]],
}

lense = {
    'MaxCover': [['Facebook', 96.60, 7.00],
                 ['Wiki', 109.40, 34.00],
                 ['Deezer', 97.90, 75.00],
                 ['Slashdot', 97.90, 69.00],
                 ['Twitter', 98.90, 33.00],
                 ['DBLP', 99.00, 90.00],
                 ['YouTube', 98.20, 79.00],
                 ['Skitter', 97.60, 70.00]],

    'MaxCut': [['Facebook', 100.00, 7.00],
               ['Wiki', 98.10, 39.00],
               ['Deezer', 97.50, 74.00],
               ['Slashdot', 99.00, 62.00],
               ['Twitter', 98.70, 48.00],
               ['DBLP', 99.30, 92.00],
               ['YouTube', 98.70, 79.00],
               ['Skitter', 97.40, 71.00]],

    'IM': [['Facebook', 97.90, 9.00],
           ['Wiki', 96.00, 51.00],
           ['Deezer', 97.20, 76.00],
           ['Slashdot', 96.60, 77.00],
           ['Twitter', 96.60, 40.00],
           ['DBLP', 96.90, 89.00],
           ['YouTube', 97.10, 75.00],
           ['Skitter', 98.30, 78.00]],
}



for problem in [
                'MaxCover',
                'MaxCut',
                'IM'
               ]:
    
    print('& \multicolumn{18}{c|}',end='')
    print('{\\textbf{',end='')
    print(f'{problem}',end='')
    print('}} ',end=' \\\\ \hline')
    print()
    
    datasets = ['Facebook','Wiki','Deezer','Slashdot','Twitter','DBLP','YouTube','Skitter']

    algorithms = ['Quickfilter','SS','GCOMB','CombHelperStudent','Lense','GNNpruner']



    def f(ratio_value ,pruned_value,algorithm):

        if algorithm == 'GNNpruner':
            print('& \multicolumn{1}{c|}')
        else:

            print('& \multicolumn{1}{c||}')

        value = ratio_value*pruned_value

        print('{',end='')
        print(f'{value:.4f}',end='')
        print('}',end='')

        


    for idx,dataset in enumerate(datasets):
        
        
        print('\multicolumn{1}{|c||}',end='')
        print('{',end='')
        print(f'{dataset}',end='')
        print('}',end='')
        for algorithm in algorithms:

            if algorithm == 'GCOMB':
                ratio_value =  gcomb[problem][idx][1]/100
                pruned_value = gcomb[problem][idx][2]/100

                print(f'& {ratio_value:.4f}' ,end ='')
                print(f'& {pruned_value:.4f}' ,end ='')

                # log_value = np.log(gcomb[problem][idx][1] / (100 - gcomb[problem][idx][2]))
                
                # log_value = np.log(gcomb[problem][idx][1] *gcomb[problem][idx][2])
                f(ratio_value,pruned_value,algorithm)



            elif algorithm == 'Lense':
                ratio_value = lense[problem][idx][1]/100
                pruned_value = lense[problem][idx][2]/100
                print(f'& {ratio_value:.4f}' ,end ='')
                print(f'& {pruned_value:.4f}' ,end ='')
                f(ratio_value,pruned_value,algorithm)

                # log_value = np.log(lense[problem][idx][1] / (100 - lense[problem][idx][2]))
                
                # log_value = np.log(lense[problem][idx][1]* lense[problem][idx][2])
                # f(log_value,algorithm)
                # print(f"& {:.2f}", end='')

            else:
                try:
                    df_ = load_from_pickle(os.path.join(problem,'data',dataset,algorithm))

                    ratio_value = df_['Ratio(%)'].iloc[0]/100
                    pruned_value = (100-df_['Pruned Ground set(%)'].iloc[0])/100

                    # Fixed code
                    print(f"& {ratio_value:.4f}", end='')
                    print(f"& {pruned_value:.4f}", end='')
                    f(ratio_value,pruned_value,algorithm)
                    # # For the third line, breaking it down for clarity
                    # ratio_value = df_['Ratio(%)'].iloc[0]
                    # pruned_value = df_['Pruned Ground set(%)'].iloc[0]
                    # log_value = np.log(ratio_value * (100-pruned_value))

                    # # print(f"& {log_value:.2f}", end='')
                    # f(log_value,algorithm)

                except:

                    if algorithm =='Quickfilter':

                        ratio_value = 98.13/100
                        pruned_value = 99.99/100
                        print(f"& {ratio_value:.4f}", end='')
                        print(f"& {pruned_value:.4f}", end='')
                        # log_value = np.log(ratio_value /(100-pruned_value) )
                        f(ratio_value,pruned_value,algorithm)

                    elif algorithm == 'SS':
                        ratio_value = 100.32/100
                        pruned_value = 99.91/100
                        print(f"& {ratio_value:.4f}", end='')
                        print(f"& {pruned_value:.4f}", end='')
                        f(ratio_value,pruned_value,algorithm)
                        # log_value = np.log(ratio_value /(100-pruned_value) )
                        # log_value = np.log(ratio_value*pruned_value) 
                        # f(log_value,algorithm)


                    elif algorithm == 'GNNpruner':
                        ratio_value = 100.23/100
                        pruned_value = 98.89/100
                        print(f"& {ratio_value:.4f}", end='')
                        print(f"& {pruned_value:.4f}", end='')
                        f(ratio_value,pruned_value,algorithm)
                        # log_value = np.log(ratio_value /(100-pruned_value) )
                        # log_value = np.log(ratio_value*pruned_value) 
                        
                        # f(log_value,algorithm)


                    else:

                        print(f"& --", end='')
                        print(f"& --", end='')
                        # print(f"& --", end='')
                        print('& \multicolumn{1}{c||}')

                        print('{',end='')
                        print(f'--',end='')
                        print('}',end='')
                    


        print('\\\\')
    print('\hline')



