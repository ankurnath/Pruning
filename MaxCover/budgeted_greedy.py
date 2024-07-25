from argparse import ArgumentParser
from utils import *
import pandas as pd
from collections import defaultdict
import numpy as np
import os

from greedy import get_gains, gain_adjustment


def modified_greedy(graph,budget,ground_set=None , node_weights= None):
    
    number_of_queries=0

    gains=get_gains(graph,ground_set)

    # get max singleton

    max_singleton = None

    max_singleton_gain = 0

    number_of_queries += len(gains)
    for element in gains:
        if node_weights[element]<= budget and gains[element] > max_singleton_gain:
            max_singleton = element
            max_singleton_gain = gains [element]


    
    


    solution=[]
    uncovered=defaultdict(lambda: True)

    constraint = 0

    N = len (gains)

    

    # for i in range(N):
    while gains:

    # for i in range(budget):
        number_of_queries+= len(gains)

        # selected_element = max(gains, key=gains.get)

        max_gain_ratio = 0

        selected_element = None

        for element in gains:
            if gains[element]/node_weights[element]> max_gain_ratio:
                max_gain_ratio = gains[element]/node_weights[element]
                selected_element = element



        
        # print (gains[selected_element])
        if max_gain_ratio == 0 :
            # print('All elements are already covered')
            break

        if node_weights[selected_element]+constraint <= budget:
            # print(node_weights[selected_element]+constraint)
            solution.append(selected_element)
            gain_adjustment(graph,gains,selected_element,uncovered)
            constraint += node_weights[selected_element]

        gains.pop(selected_element)

    # print('Number of queries:',number_of_queries)
        
    if calculate_cover(graph,solution)< max_singleton_gain :
        solution = [ max_singleton ]

    return solution,number_of_queries

        


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument( "--dataset", type=str, default='Facebook', help="Name of the dataset to be used (default: 'Facebook')" )
    parser.add_argument( "--budgets", nargs='+', type=int, help="Budgets" )
    parser.add_argument("--cost_model",type= str, default= 'random', help = 'model of node weights')
  
    args = parser.parse_args()

    file_path=f'../../data/test/{args.dataset}'

    # file_path=f'../../data/s/{args.dataset}'
    graph=load_from_pickle(file_path)

    node_weights = load_from_pickle(f'../../data/test/{args.dataset}_weights_{args.cost_model}')
    # load_graph_file_path=f'../../data/snap_dataset/{args.dataset}.txt'
    # graph=nx.read_edgelist(f'../../data/snap_dataset/{args.dataset}.txt', create_using=nx.Graph(), nodetype=int)

    
    df=defaultdict(list)

    for budget in args.budgets:

        solution,_= modified_greedy(graph=graph,budget=budget,node_weights=node_weights)

        subgraph =make_subgraph(graph,solution)
        Pv=1-subgraph.number_of_nodes()/graph.number_of_nodes()
        Pe=1-subgraph.number_of_edges()/graph.number_of_edges()

        cover=calculate_cover(graph,solution)

        df['Budget'].append(budget)
        df['Pv'].append(Pv)
        df['Pe'].append(Pe)
        df['Objective Value'].append(cover)
        df['Objective Value (Ratio)'].append(cover/graph.number_of_nodes())
        df['Solution'].append(solution)
        df['Cost'].append([node_weights[node] for node in solution])
        
        
    df=pd.DataFrame(df)
    print(df)
    save_folder=f'data/{args.dataset}'
    file_path=os.path.join(save_folder,'Greedy')
    os.makedirs(save_folder,exist_ok=True)
    save_to_pickle(df,file_path)








