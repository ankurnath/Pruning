import pandas as pd
import os
from collections import defaultdict
import pickle
# gaint_df = [] 
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
    root_folder=os.path.join(problem,'data')
    
    
    datasets=os.listdir(root_folder)
    print(datasets)
    datasets = ['Facebook','Wiki','Deezer','Slashdot','Twitter','DBLP','YouTube','Skitter']
    for dataset in datasets:
        print('*'*20)
        print(dataset)
        dataset_path = os.path.join(root_folder,dataset)
        algorthims = os.listdir(dataset_path)

        # df ={'algorithm':[],'Size of Ground Set':[],'Ratio':[],'Queries':[]}
        df = defaultdict(list)
        # for algorthim in ['Quickfilter','SS','LeNSE','CombHelperTeacher','CombHelperStudent','GNNpruner']:
        for algorthim in [
           'Quickfilter',
        #    'SS',
        #    'CombHelperStudent',
        #    'GNNpruner'
           ]:
          try:
            df_ = load_from_pickle(os.path.join(dataset_path,algorthim))

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

          except:
            pass

        df = pd.DataFrame(df)

        df['Size of Ground Set']=100-df['Size of Ground Set'].round(4)


        print(df)
