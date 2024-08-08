from argparse import ArgumentParser
from utils import load_from_pickle,calculate_cover,save_to_pickle,make_subgraph
import pandas as pd
from collections import defaultdict
from greedy import greedy
import numpy as np
import os 


def test(dataset,budgets):

    load_graph_file_path=f'../../data/test/{args.dataset}'

    graph=load_from_pickle(load_graph_file_path)
    N=graph.number_of_nodes() 
    outdegree = [(node, graph.degree(node)) for node in graph.nodes()]
    outdegree=sorted(outdegree,key=lambda x:x[1],reverse=True)

    dict=load_from_pickle(f'pretrained agents/degree/{dataset}')

    df=defaultdict(list)

    for budget in budgets:
        percentile=np.interp(budget/N,dict['Budgets'],dict['Ranks'])
        n = int(N* percentile)
        pruned_universe=[outdegree[i][0] for i in range(n)]
        # print(pruned_universe)
        subgraph =make_subgraph(graph,pruned_universe)

        print('*'*20)
        print('Budget:',budget)
        print('Percentile:',percentile)
        print('Core Size:',n)
        Pv=1-subgraph.number_of_nodes()/graph.number_of_nodes()
        Pe=1-subgraph.number_of_edges()/graph.number_of_edges()
        print('Pv:',Pv)
        print('Pe:',Pe)
        solution_subgraph = greedy(subgraph,budget,pruned_universe)

        coverage= calculate_cover(graph,solution_subgraph)
        # print('Coverage:',coverage/N)

        df['Dataset'].append(args.dataset)
        
        df['Pv'].append(Pv)
        df['Pe'].append(Pe)
        df['Pruned Ground Set (Ratio)'].append(len(pruned_universe)/N)
        df['Budget'].append(budget)
        df['Dataset Path'].append(load_graph_file_path)
        
        
        df['Solution'].append(solution_subgraph)
        df['Objective Value'].append(coverage)
        df['Objective Value (Ratio)'].append(coverage/graph.number_of_nodes())

        
        print('*'*20)

    df=pd.DataFrame(df)

    print(df)

    save_folder='data/Degree'
    file_path=os.path.join(save_folder,args.dataset)
    os.makedirs(save_folder,exist_ok=True)
    save_to_pickle(df,file_path) 


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument( "--dataset", type=str, default='Facebook', help="Name of the dataset to be used for training (default: 'Facebook')" )
    parser.add_argument( "--budgets", nargs='+', type=int, help="Budgets" )

    
    


    args = parser.parse_args()
    test(args.dataset,args.budgets)