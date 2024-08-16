from argparse import ArgumentParser
from utils import *
import pandas as pd
from collections import defaultdict
from greedy import greedy
import numpy as np
import os 


def test(dataset,budget):

    sprint(dataset)
    sprint(budget)


    load_graph_file_path=f'../../data/snap_dataset/{dataset}.txt'
    graph = load_graph(load_graph_file_path)


    
    

    dict=load_from_pickle(f'pretrained agents/GCOMB/{dataset}')
    start = time.time()
    outdegree = [(node, graph.degree(node)) for node in graph.nodes()]
    outdegree=sorted(outdegree,key=lambda x:x[1],reverse=True)
    N = graph.number_of_nodes()
    percentile=np.interp(budget/N,dict['Budgets'],dict['Ranks'])
    n = int(N* percentile)
    pruned_universe=[outdegree[i][0] for i in range(n)]
    end= time.time()

    time_to_prune = end-start

    print('time elapsed to pruned',time_to_prune)

    ##################################################################

    Pg=len(pruned_universe)/graph.number_of_nodes()
    start = time.time()
    solution_unpruned,queries_unpruned= greedy(graph,budget)
    end = time.time()
    time_unpruned = round(end-start,4)
    print('Elapsed time (unpruned):',round(time_unpruned,4))

    start = time.time()
    solution_pruned,queries_pruned = greedy(graph=graph,budget=budget,ground_set=pruned_universe)
    end = time.time()
    time_pruned = round(end-start,4)
    print('Elapsed time (pruned):',time_pruned)
    
    
    objective_unpruned = calculate_cover(graph,solution_unpruned)
    objective_pruned = calculate_cover(graph,solution_pruned)
    
    ratio = objective_pruned/objective_unpruned


    print('Performance of GCOMB')
    print('Size Constraint,k:',budget)
    print('Size of Ground Set,|U|:',graph.number_of_nodes())
    print('Size of Pruned Ground Set, |Upruned|:', len(pruned_universe))
    print('Pg(%):', round(Pg,4)*100)
    print('Ratio:',round(ratio,4)*100)
    print('Queries:',round(queries_pruned/queries_unpruned,4)*100)


    save_folder = f'data/{dataset}'
    os.makedirs(save_folder,exist_ok=True)
    save_file_path = os.path.join(save_folder,'GCOMB')

    df ={      'Dataset':dataset,'Budget':budget,'Objective Value(Unpruned)':objective_unpruned,
              'Objective Value(Pruned)':objective_pruned ,'Ground Set': graph.number_of_nodes(),
              'Ground set(Pruned)':len(pruned_universe), 'Queries(Unpruned)': queries_unpruned,'Time(Unpruned)':time_unpruned,
              'Time(Pruned)': time_pruned,
              'Queries(Pruned)': queries_pruned, 'Pruned Ground set(%)': round(Pg,4)*100,
              'Ratio(%)':round(ratio,4)*100, 'Queries(%)': round(queries_pruned/queries_unpruned,4)*100,
              'TimeRatio': time_pruned/time_unpruned,
              'TimeToPrune':time_to_prune

              }

   
    df = pd.DataFrame(df,index=[0])
    save_to_pickle(df,save_file_path)
    print(df)

    ###################################################################################################
        

if __name__ == "__main__":


    parser = ArgumentParser()
    parser.add_argument( "--dataset", type=str, default='Facebook', help="Name of the dataset to be used for training (default: 'Facebook')" )
    parser.add_argument("--budget", type=int,required=True,default=100, help="Budgets")
    

    args = parser.parse_args()
    test(dataset=args.dataset,budget=args.budget)