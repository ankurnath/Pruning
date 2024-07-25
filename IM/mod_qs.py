from argparse import ArgumentParser
from utils import *
import pandas as pd
from collections import defaultdict
import numpy as np
import random
import os

from greedy import imm





def QS(dataset,budgets,num_rr,delta,seed):

    graph = load_from_pickle(file_path=f'../../data/test/{dataset}')
    nodes = list(graph.nodes()) 

    graph_,_ = get_graph(graph)

    # del graph

    RR = []
    worker = []
    worker_num =NUM_PROCESSORS
    create_worker(num =worker_num, worker = worker, node_num = None, model = 'IC', graph_=graph_)

    for ii in range(worker_num):
            worker[ii].inQ.put(num_rr / worker_num)
    for w in worker:
        R_list = w.outQ.get()
        RR += R_list

    finish_worker(worker)
    pruned_universe = []
    rr_degree=defaultdict(int)
    node_rr_set = defaultdict(list)


    covered_rr_set=set()
    for j,rr in enumerate(RR):
        for rr_node in rr:
            rr_degree[rr_node]+=1
            node_rr_set[rr_node].append(j)


    curr_obj = 0
    for node in rr_degree:
        if rr_degree[node]>=delta*curr_obj/budgets:
            pruned_universe.append (node)
            curr_obj += rr_degree[node]
            

            for index in node_rr_set[node]:
                if index not in covered_rr_set:
                    covered_rr_set.add(index)
                    for rr_node in RR[index]:
                        rr_degree[rr_node]-=1

   
    
    # print(len(pruned_universe))
    print("Only taking outgoing edges from the ground set. Ask Professor for clarification")    
    subgraph = make_subgraph(graph,pruned_universe)

    _,_,solution,_ = imm (graph=graph,seed_size=budgets,seed=seed)
    print("Whole graph:",[graph.degree(node) for node in solution])

    _,_,pruned_solution,_ = imm (graph=subgraph,seed_size=budgets,seed=seed)
    print("Subgraph:",[graph.degree(node) for node in pruned_solution])

    folder_path = f'../../data/test/{dataset}_subgraphs'


    pruned_spread = calculate_spread(folder_path,pruned_solution)

    spread = calculate_spread(folder_path,solution)

    print(f"Ratio:{pruned_spread/spread}")





if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument( "--dataset", type=str, default='Facebook_CONST', help="Name of the dataset to be used (default: 'Facebook')" )
    parser.add_argument("--delta",type=float,default=0.1,help = "Delta")
    # parser.add_argument( "--model", type=str, default='CONST', help="Name of the dataset to be used (default: 'Facebook')" )
    # parser.add_argument("--budgets", nargs='+', type=int, help="Budgets")
    parser.add_argument("--budget", type=int, default= 10  , help="Budget")
    parser.add_argument("--num_rr", type=int, default= 10000  , help="Budgets")
    # parser.add_argument( "--budgets", type=int, default=10 , help="Budgets" )
    parser.add_argument( "--seed", type=int, default=0, help="Random seed for reproducibility (default: 0)" )

    args = parser.parse_args()


    # graph = load_from_pickle(file_path=f'../../data/test/{args.dataset}')

    
    # print([graph.degree(node) for node in solution])


    QS(dataset=args.dataset,budgets=args.budget,num_rr=args.num_rr,delta= args.delta,seed=args.seed)





    








