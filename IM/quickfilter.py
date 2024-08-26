from argparse import ArgumentParser
from utils import *
import pandas as pd
from collections import defaultdict
import numpy as np
import random
import os
from tqdm import tqdm
from imm import imm


def bfs(subgraph, initial_nodes,graph):
    activated_nodes = set(initial_nodes)
    queue = deque(initial_nodes)
    

    while queue:
        node = queue.popleft()

        for neighbor in graph.neighbors(node):
            if (node,neighbor) in subgraph and neighbor not in activated_nodes:
                activated_nodes.add(neighbor)
                queue.append(neighbor)

    return len(activated_nodes) 


def calculate_spread(subgraphs,solution,graph):

    totol_spread = 0

    for i in range(len(subgraphs)):
       totol_spread += bfs(subgraphs[i],solution,graph)


    
    

    # step_size = 20
    # for i in range(0,args.mc,step_size):
    #     processes=[]
    # for i in range(0,len(subgraphs),step_size):
    #     arguments=[]
    #     for _ in range(i,min(i+step_size,len(subgraphs))):
    #         arguments.append ((subgraphs[i],solution,graph))
    #     with Pool() as pool:
    #         totol_spread += np.sum(pool.starmap(bfs,arguments))


    return totol_spread/len(subgraphs)



def quickfilter(dataset,budgets,delta=0.1):

    

    load_graph_file_path=f'../../data/test/{dataset}'
    graph = load_from_pickle(load_graph_file_path)


    load_subgrapgh_file_path = f'../../data/test/{dataset}_subgraphs'
    subgraphs = []

    for file in os.listdir(load_subgrapgh_file_path)[:5]:
        file_path = os.path.join(load_subgrapgh_file_path,file)
        subgraphs.append(load_from_pickle(file_path,quiet=True))


    start =time.time()
    for budget in budgets:

        curr_obj=0

        pruned_universe=[]

        # for node in graph.nodes():
        for node in tqdm(graph.nodes()):

            # step_size = 20

            # new_solution = pruned_universe+[node]
            # totol_spread = 0
            # for i in range(0,len(subgraphs),step_size):
            #     arguments=[]
            #     for _ in range(i,min(i+step_size,len(subgraphs))):
            #         arguments.append ((subgraphs[i],new_solution,graph))
            #     with Pool() as pool:
            #         totol_spread += np.sum(pool.starmap(bfs,arguments))
            
            # spread=totol_spread/len(subgraphs)
            
            
            # gain = spread - curr_obj
            gain = calculate_spread(subgraphs=subgraphs,solution= pruned_universe+[node],graph=graph) -curr_obj
            if gain >= delta/budget*curr_obj:
                curr_obj+= gain
                pruned_universe.append(node)
        # break


    print(curr_obj)

    print('Size of Pruned Universe:',len(pruned_universe))

    end = time.time()

    total_time = (end-start)/60
    print('Elapesd time:',total_time)

    # print(len(pruned_universe))
    print("Only taking outgoing edges from the ground set. Ask Professor for clarification")    
    subgraph = make_subgraph(graph,pruned_universe)

    _,_,solution,_ = imm (graph=graph,seed_size=budgets[0],seed=0)
    print("Whole graph:",[graph.degree(node) for node in solution])

    _,_,pruned_solution,_ = imm (graph=subgraph,seed_size=budgets[0],seed=0)
    print("Subgraph:",[graph.degree(node) for node in pruned_solution])

    folder_path = f'../../data/test/{dataset}_subgraphs'


    pruned_spread = calculate_spread(folder_path,pruned_solution)

    spread = calculate_spread(folder_path,solution)

    print(f"Ratio:{pruned_spread/spread}")


if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument( "--dataset", type=str, default='Facebook_CONST', help="Name of the dataset to be used (default: 'Facebook')" )
    # parser.add_argument( "--model", type=str, default='CONST', help="Name of the dataset to be used (default: 'Facebook')" )
    parser.add_argument("--budgets", nargs='+', type=int, help="Budgets")
    # parser.add_argument( "--budgets", type=int, default=10 , help="Budgets" )
    parser.add_argument( "--seed", type=int, default=0, help="Random seed for reproducibility (default: 0)" )

    args = parser.parse_args()


    # graph = load_from_pickle(file_path=f'../../data/test/{args.dataset}')

    
    # print([graph.degree(node) for node in solution])


    quickfilter(dataset=args.dataset,budgets=args.budgets)





    








