from argparse import ArgumentParser
from utils import *
import pandas as pd
from collections import defaultdict
import numpy as np
import random
import os

from imm import * 

from greedy import *





# def QS(dataset,budget,num_rr,delta,seed):
def quickfilter(dataset,budget,delta,num_rr,seed,eps=0.1):

    
    load_graph_file_path=f'../../data/snap_dataset/{dataset}.txt'

    graph = load_graph(load_graph_file_path)


    

    # sprint(calculate_spread(graph=graph,solution=sorted([graph.degree(node) for node in graph.nodes()])[::-1][:20]))

    start = time.time()
    # gains = get_gains(graph,ground_set=None)
    gains,node_rr_set,RR = get_gains(graph,num_rr)
    curr_obj = 0
    queries_to_prune = 0
    # pruned_universe=[] 
    a = set()
    # a_start = set() 
    a_start = np.argmax(gains)
    a_s = set()
    covered_rr_set = set ()

    obj_a_s = 0
    uncovered=defaultdict(lambda: True)

    N = graph.number_of_nodes()
    for node in tqdm(graph.nodes()):
        queries_to_prune += 1
        if gains[node]>= delta/budget*curr_obj:
            curr_obj+=gains[node]
            # pruned_universe.append(node)
            a.add(node)
            # gain_adjustment(graph,gains,node,uncovered)
            gain_adjustment(gains=gains,node_rr_set=node_rr_set,
                            RR=RR,selected_element=node,covered_rr_set=covered_rr_set)


        ### New addition
        if curr_obj > N/eps*obj_a_s:
            # print('This happened')
            
            # a = a.difference(a_s)
            a.difference_update(a_s)
            a_s = a.copy()

            obj_a_s = calculate_spread(graph=graph,solution=a_s)
            curr_obj = calculate_spread(graph=graph,solution=a)
            queries_to_prune +=2
            
    end= time.time()

    time_to_prune = end-start

    print('time elapsed to pruned',time_to_prune)
    a.add(a_start)
    pruned_universe = list(a)
    
    


    ##################################################################

    Pg=len(pruned_universe)/graph.number_of_nodes()
    start = time.time()
    # solution_unpruned, _ = imm(graph=graph,seed_size=budget,seed=seed)
    queries_unpruned  = budget/2 * (2*graph.number_of_nodes() - budget +1) 
    solution_unpruned = imm(graph=graph,seed_size=budget,seed=seed)
    end = time.time()


    # sprint([graph.degree(node) for node in solution_unpruned])
    
    time_unpruned = round(end-start,4)
    print('Elapsed time (unpruned):',round(time_unpruned,4))
    
    subgraph = make_subgraph(graph,pruned_universe) 
    start = time.time()
    solution_pruned = imm(graph=subgraph,seed_size=budget, seed=seed)
    queries_pruned  = budget/2 * (2*len(pruned_universe) - budget +1) 

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

    df ={     'Dataset':dataset,'Budget':budget,
              'Delta':delta,
              'QueriesToPrune': queries_to_prune,
              'Objective Value(Unpruned)':objective_unpruned,
              'Objective Value(Pruned)':objective_pruned ,
              'Ground Set': graph.number_of_nodes(),
              'Ground set(Pruned)':len(pruned_universe), 
              'Queries(Unpruned)': queries_unpruned,'Time(Unpruned)':time_unpruned,
              'Time(Pruned)': time_pruned,
              'Queries(Pruned)': queries_pruned, 
              'Pruned Ground set(%)': round(Pg,4)*100,
              'Ratio(%)':round(ratio,4)*100, 
              'Queries(%)': round(queries_pruned/queries_unpruned,4)*100,
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
    parser.add_argument("--eps", type=float, default=0.1, help="eps")
    args = parser.parse_args()

    dataset = args.dataset
    delta = args.delta 
    budget = args.budget
    num_rr = args.num_rr
    seed = args.seed

    sprint(dataset)
    sprint(budget)
    sprint(delta)
    sprint(budget)
    sprint(seed)




    quickfilter(dataset=dataset,budget=budget,
                num_rr=num_rr,delta = delta,
                seed=seed)





    








