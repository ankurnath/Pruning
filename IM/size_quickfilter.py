from argparse import ArgumentParser
from utils import *
import pandas as pd
from collections import defaultdict
import numpy as np
import random
import os

from imm import * 

from greedy import *





def QS(dataset,budget,num_rr,delta,seed):

    
    load_graph_file_path=f'../../data/snap_dataset/{dataset}.txt'

    graph = load_graph(load_graph_file_path)


    

    # sprint(calculate_spread(graph=graph,solution=sorted([graph.degree(node) for node in graph.nodes()])[::-1][:20]))

    start = time.time()
    gains,node_rr_set,RR = get_gains(graph,num_rr)


    pruned_universe = []

    covered_rr_set = set ()

    curr_obj = 0

    for node in gains:
        if gains[node]>=delta*curr_obj/budget:
            pruned_universe.append (node)
            curr_obj += gains[node]
            gain_adjustment(gains=gains,node_rr_set=node_rr_set,RR=RR,selected_element=node,covered_rr_set=covered_rr_set)


    end= time.time()

    time_to_prune = end-start

    print('time elapsed to pruned',time_to_prune)

    # sprint([graph.degree(node) for node in pruned_universe])
   
    subgraph = make_subgraph(graph,pruned_universe)

    ##################################################################

    Pg=len(pruned_universe)/graph.number_of_nodes()
    start = time.time()
    # solution_unpruned, _ = imm(graph=graph,seed_size=budget,seed=seed)
    solution_unpruned = imm(graph=graph,seed_size=budget,seed=seed)
    end = time.time()


    # sprint([graph.degree(node) for node in solution_unpruned])
    
    time_unpruned = round(end-start,4)
    print('Elapsed time (unpruned):',round(time_unpruned,4))

    start = time.time()
    solution_pruned = imm(graph=subgraph,seed_size=budget, seed=seed)

    # sprint([graph.degree(node) for node in solution_pruned])
    end = time.time()
    time_pruned = round(end-start,4)
    print('Elapsed time (pruned):',time_pruned)

    objective_pruned = calculate_spread(graph=graph,solution=solution_pruned)
    objective_unpruned = calculate_spread(graph=graph,solution=solution_unpruned)

    sprint(objective_pruned)
    sprint(objective_unpruned)
    ratio = objective_pruned/objective_unpruned


    print('Performance of QuickFilter')
    print('Size Constraint,k:',budget)
    print('Size of Ground Set,|U|:',graph.number_of_nodes())
    print('Size of Pruned Ground Set, |Upruned|:', len(pruned_universe))
    print('Pg(%):', round(Pg,4)*100)
    print('Ratio:',round(ratio,4)*100)


    save_folder = f'data/{dataset}'
    os.makedirs(save_folder,exist_ok=True)
    save_file_path = os.path.join(save_folder,'Quickfilter')

    df ={     'Dataset':dataset,'Budget':budget,'Delta':delta,'Objective Value(Unpruned)':objective_unpruned,
              'Objective Value(Pruned)':objective_pruned ,
              'Ground Set': graph.number_of_nodes(),
              'Ground set(Pruned)':len(pruned_universe), 
            #   'Queries(Unpruned)': queries_unpruned,
              'Time(Unpruned)':time_unpruned,
              'Time(Pruned)': time_pruned,
            #   'Queries(Pruned)': queries_pruned, 
              'Pruned Ground set(%)': round(Pg,4)*100,
              'Ratio(%)':round(ratio,4)*100, 
            #   'Queries(%)': round(queries_pruned/queries_unpruned,4)*100,
              'TimeRatio': time_pruned/time_unpruned,
              'TimeToPrune':time_to_prune

              }

   
    df = pd.DataFrame(df,index=[0])
    save_to_pickle(df,save_file_path)
    print(df)

    ###################################################################################################

    





if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument( "--dataset", type=str, default='Facebook', help="Name of the dataset to be used (default: 'Facebook')" )
    parser.add_argument("--delta",type=float,default=0.1,help = "Delta")
    # parser.add_argument( "--model", type=str, default='CONST', help="Name of the dataset to be used (default: 'Facebook')" )
    # parser.add_argument("--budgets", nargs='+', type=int, help="Budgets")
    parser.add_argument("--budget", type=int, default= 100  , help="Budget")
    parser.add_argument("--num_rr", type=int, default= 100000  , help="Number of RR sets")
    # parser.add_argument( "--budgets", type=int, default=10 , help="Budgets" )
    parser.add_argument( "--seed", type=int, default=0, help="Random seed for reproducibility (default: 0)" )

    args = parser.parse_args()


    # graph = load_from_pickle(file_path=f'../../data/test/{args.dataset}')

    
    # print([graph.degree(node) for node in solution])


    QS(dataset=args.dataset,budget=args.budget,num_rr=args.num_rr,delta= args.delta,seed=args.seed)





    








