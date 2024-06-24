from argparse import ArgumentParser
from utils import load_from_pickle,calculate_cut,save_to_pickle,make_subgraph
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

    dict=load_from_pickle(f'pretrained agents/GCOMB/{dataset}')

    for budget in budgets:
        percentile=np.interp(budget/N,dict['Budgets'],dict['Ranks'])
        n = int(N* percentile)
        pruned_universe=outdegree[:n]
        subgraph =make_subgraph(graph,pruned_universe)

        print('*'*20)
        print('Budget:',budget)
        print('Percentile:',percentile)
        print('Core Size:',n)
        print('Pv:',1-subgraph.number_of_nodes()/graph.number_of_nodes())
        print('Pe:',1-subgraph.number_of_edges()/graph.number_of_edges())
        solution_subgraph = greedy(subgraph,budget)

        cut= calculate_cut(graph,solution_subgraph)
        print('Cut:',cut/graph.number_of_edges())
        print('*'*20)

        


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
    test(args.dataset,args.budgets)