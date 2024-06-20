from argparse import ArgumentParser
from utils import load_from_pickle,calculate_cover,save_to_pickle
import pandas as pd
from collections import defaultdict
from greedy import greedy
import numpy as np
import os 


def train(dataset,budgets):

    load_graph_file_path=f'../../data/train/{args.dataset}'

    graph=load_from_pickle(load_graph_file_path)

    max_budget=max(budgets)

    outdegrees=[(node,graph.degree(node)) for node in graph.nodes()]
    outdegrees=sorted(outdegrees,key=lambda x:x[1],reverse=True)
    ranks={}
    for i,(node,_) in enumerate(outdegrees):
        ranks[node] = i+1

    greedy_solution=greedy(graph,max_budget)

    r_temp=[ranks[node] for node in greedy_solution]
    

    r=[None]*len(budgets)

    for i,budget in enumerate(budgets):
        
        r[i]=max(r_temp[:budget])

    budgets=np.array(budgets)/graph.number_of_nodes()
    r=np.array(r)/graph.number_of_nodes()

    print('Budgets:',budgets)
    print('Ranks:',r)

    save_folder='pretrained agents/degree'
    os.makedirs(save_folder,exist_ok=True)

    df={
        'Dataset': f'{dataset}',
        'Path':load_graph_file_path,
        'Budgets':budgets,
        'Ranks':r,
        }
    
    # print(df)
    file_path=os.path.join(save_folder,f'{dataset}')
    save_to_pickle(df,file_path)
    # df.to_pickle(file_path)

    

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        default='Facebook',
        help="Name of the dataset to be used for training (default: 'Facebook')"
    )
    parser.add_argument(
        "--budgets",
        nargs='+',
        type=int,
        help="Budgets"
    )
    


    args = parser.parse_args()
    train(args.dataset,args.budgets)

   








