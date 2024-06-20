from argparse import ArgumentParser
from utils import load_from_pickle,calculate_cover,save_to_pickle
import pandas as pd
from collections import defaultdict
from greedy import prob_greedy
import numpy as np
import os 


def train(dataset,budgets,m,delta):

    load_graph_file_path=f'../../data/train/{args.dataset}'

    graph=load_from_pickle(load_graph_file_path)

    max_budget=max(budgets)

    outdegrees=[(node,graph.degree(node)) for node in graph.nodes()]
    outdegrees=sorted(outdegrees,key=lambda x:x[1],reverse=True)
    ranks={}
    for i,(node,_) in enumerate(outdegrees):
        ranks[node] = i+1
    r=[0]*len(budgets)
    for _ in range(m):

        greedy_solution=prob_greedy(graph,max_budget,ground_set=None,delta=delta*outdegrees[0][1])
        r_temp=[ranks[node] for node in greedy_solution]
        for i,budget in enumerate(budgets):
            
            r[i]=max(r[i],max(r_temp[:budget]))

    budgets=np.array(budgets)/graph.number_of_nodes()
    r=np.array(r)/graph.number_of_nodes()

    print('Budgets:',budgets)
    print('Ranks:',r)

    save_folder='pretrained agents/GCOMB'
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

    parser.add_argument(
        "--delta",
        type=float,
        default=0.001,
        help="Delta"
    )

    parser.add_argument(
        "--m",
        type=int,
        default=30,
        help='--number_of_attempts'

    )
    
    args = parser.parse_args()
    train(args.dataset,args.budgets,args.m,args.delta)

   








