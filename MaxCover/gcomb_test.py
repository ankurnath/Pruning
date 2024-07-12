from argparse import ArgumentParser
from utils import *
import pandas as pd
from collections import defaultdict
from greedy import greedy
import numpy as np
import os 


def test(dataset,budgets,test_on_whole_dataset = False):


    if test_on_whole_dataset:
        load_graph_file_path=f'../../data/snap_dataset/{args.dataset}.txt'
        graph=nx.read_edgelist(f'../../data/snap_dataset/{args.dataset}.txt', create_using=nx.Graph(), nodetype=int)

    else:
        load_graph_file_path=f'../../data/test/{args.dataset}'

        graph=load_from_pickle(load_graph_file_path)
    N=graph.number_of_nodes() 
    print('Number of Nodes in the whole graph',N)
    outdegree = [(node, graph.degree(node)) for node in graph.nodes()]
    outdegree=sorted(outdegree,key=lambda x:x[1],reverse=True)

    dict=load_from_pickle(f'pretrained agents/GCOMB/{dataset}')

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
        solution_subgraph,number_of_queries_pruned = greedy(subgraph,budget,pruned_universe)


        solution_greedy,number_of_queries_greedy = greedy(subgraph,budget)

        

        coverage= calculate_cover(graph,solution_subgraph)
        greedy_coverage= calculate_cover(graph,solution_greedy)

        print('Value Reduced:',(coverage-greedy_coverage)/greedy_coverage*100)
        print('Queries Reduced:',(number_of_queries_pruned-number_of_queries_greedy)/number_of_queries_greedy*100)

        print('Coverage:',coverage/N)

        df['Dataset'].append(args.dataset)
        df['Pruned Ground Set (Ratio)'].append(len(pruned_universe)/N)
        df['Pv'].append(Pv)
        df['Pe'].append(Pe)
        df['Budget'].append(budget)
        df['Dataset Path'].append(load_graph_file_path)
        
        
        df['Solution'].append(solution_subgraph)
        df['Objective Value'].append(coverage)
        df['Objective Value (Ratio)'].append(coverage/graph.number_of_nodes())

        
        print('*'*20)

    df=pd.DataFrame(df)

    print(df)

    save_folder='data/GCOMB'
    file_path=os.path.join(save_folder,args.dataset)
    os.makedirs(save_folder,exist_ok=True)
    save_to_pickle(df,file_path) 



    # for budget in budgets:
    #     percentile=np.interp(budget/N,dict['Budgets'],dict['Ranks'])
    #     n = int(N* percentile)
    #     pruned_universe=outdegree[:n]
    #     subgraph =make_subgraph(graph,pruned_universe)

    #     print('*'*20)
    #     print('Budget:',budget)
    #     print('Percentile:',percentile)
    #     print('Core Size:',n)
    #     print('Pv:',1-subgraph.number_of_nodes()/graph.number_of_nodes())
    #     print('Pe:',1-subgraph.number_of_edges()/graph.number_of_edges())
    #     solution_subgraph = greedy(subgraph,budget)

    #     coverage= calculate_cover(graph,solution_subgraph)
    #     print('Coverage:',coverage/N)
    #     print('*'*20)

        


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
    test(args.dataset,args.budgets,True)