from argparse import ArgumentParser
from utils import *
import pandas as pd
from collections import defaultdict
import numpy as np
import random
import os
import matplotlib.pyplot as plt

def imm(graph, seed_size, model="IC", epsoid=0.5, l=1,seed=0):
    """
    graph must be a file path to a .txt file of edge lists where the first line has the number of nodes in the first
    column, or it must be a networkx graph object with edge weights under the key 'weight'.
    """
    graph_, node_num = get_graph(graph)

    # np.random.seed(args.seed)
    # random.seed(args.seed)
    worker = []
    n = node_num
    k = seed_size
    l = l * (1 + math.log(2) / math.log(n))
    R = sampling(epsoid, l, node_num, seed_size, worker, graph_, model)
    Sk, z, x,merginal_gains = node_selection(R, k, node_num)
    # print(R[:10])
    return Sk, z,x,merginal_gains



 

        


if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument( "--dataset", type=str, default='Facebook', help="Name of the dataset to be used (default: 'Facebook')" )
    # parser.add_argument( "--model", type=str, default='CONST', help="Name of the dataset to be used (default: 'Facebook')" )
    parser.add_argument( "--budgets", type=int, default=10 , help="Budgets" )
    parser.add_argument( "--seed", type=int, default=0, help="Random seed for reproducibility (default: 0)" )

    args = parser.parse_args()


    graph = load_from_pickle(file_path=f'../../data/test/{args.dataset}')

    # load_graph_file_path=f'../../data/snap_dataset/{args.dataset}.txt'
    # graph=nx.read_edgelist(f'../../data/snap_dataset/{args.dataset}.txt', create_using=nx.Graph(), nodetype=int)

    # folder_path = f'../../data/train/{args.dataset}_subgraphs'

    # spreads = []
    # for budget in range(1,args.budgets+1):
    #     s = 0
    #     rep = 1
    #     for _ in range(rep):
    #         _,_,solution,_ =imm (graph=graph,seed_size=budget,seed=args.seed)
    #         s += calculate_spread(folder_path,solution)
    #     spreads.append(s/rep)

    # # _,_,solution,_ =imm (graph=graph,seed_size=args.budgets,seed=args.seed)

    # # print(solution)
    # # Set the figure DPI for better quality
    # plt.figure(dpi=200)

    # # Plot the data with markers
    # plt.plot(range(1, args.budgets + 1), spreads, marker='o')

    # # Add labels and title
    # plt.xlabel('Budget')
    # plt.ylabel('Expected Spread (10000 MC)')
    # plt.title('Facebook (Train)')

    # # Display the plot
    # plt.show()

    # print (calculate_spread(folder_path,solution))
    # for budget in args.budgets:

    print('Number of nodes:',graph.number_of_nodes())
    degree = sorted([graph.degree(node) for node in graph.nodes()],reverse=True)
    print('Sorted Degree', degree[:20])
    _,_,solution,_ =imm (graph=graph,seed_size=args.budgets,seed=args.seed)
    print(solution)
    print([graph.degree(node) for node in solution])


    





    








