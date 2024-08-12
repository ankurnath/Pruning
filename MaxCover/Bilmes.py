from argparse import ArgumentParser
from utils import *
import pandas as pd
from collections import defaultdict

from greedy import greedy
import numpy as np
import random
import heapq
import os
from tqdm import tqdm

# from large_graph import Graph

def SS(dataset,r,c,budget):

    # file_path=f'../../data/test/{dataset}'
    # graph=load_from_pickle(file_path)

    file_path=f'../../data/snap_dataset/{args.dataset}.txt'
    # graph=nx.read_edgelist(f'../../data/snap_dataset/{args.dataset}.txt', create_using=nx.Graph(), nodetype=int)
    # graph = Graph(file_path=file_path)
    graph = load_graph(file_path=file_path)

    pruned_universe=set()

    universe=list(graph.nodes())
    n=graph.number_of_nodes()
    while len(universe)> r*np.log2(n):
        print('Size of pruned universe:',len(universe))
        U=random.sample(universe,int(r*np.log2(n)))
        universe = set(universe)
        for node in tqdm(U):
            universe.remove(node)
        # universe = list(universe)
        U=set(U)
        pruned_universe=pruned_universe.union(U)


        universe_gain=calculate_cover(graph,universe) # f(V)

        # for v in universe:

        universe_u_gain = {} # f(V U u)
        u_gain = {} # f(u)
        # get all neighbors 
        
        
        for u in tqdm(U):
            universe.add(u)
            universe_u_gain[u] = calculate_cover (graph ,universe)
            universe.remove(u)
            u_gain[u] = calculate_cover (graph , [u])


        lst = []

        for v in tqdm(universe):

            w=float('inf')
            
            # for u in graph.neighbors(v):
                
            for u in U:
                # universe_copy=universe.copy()
                # universe_copy.append(u)
                
                local_gain = calculate_cover(graph,[u,v])-u_gain[u] # f(v U u) -f(u)
                # print(local_gain)

                global_gain = universe_u_gain[u]-universe_gain
                w=min(w,local_gain-global_gain)

            lst.append((w,v))

        remove_nodes=heapq.nsmallest(int((1-1/np.sqrt(c))*len(universe)), lst)
        # print(remove_nodes)
        universe = set(universe)
        for w,node in tqdm(remove_nodes):
            # if w>0:
            #     print(w)
            universe.remove(node)
            # universe.re
        universe = list(universe)

        

    pruned_universe=pruned_universe.union(set(universe))

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


    print('Performance of QuickFilter')
    print('Size Constraint,k:',budget)
    print('Size of Ground Set,|U|:',graph.number_of_nodes())
    print('Size of Pruned Ground Set, |Upruned|:', len(pruned_universe))
    print('Pg(%):', round(Pg,4)*100)
    print('Ratio:',round(ratio,4)*100)
    print('Queries:',round(queries_pruned/queries_unpruned,4)*100)


    save_folder = f'data/{dataset}'
    os.makedirs(save_folder,exist_ok=True)
    save_file_path = os.path.join(save_folder,'SS')

    df ={'Dataset':dataset,'Budget':budget,'r':r,'c':c,'Objective Value(Unpruned)':objective_unpruned,
              'Objective Value(Pruned)':objective_pruned ,'Ground Set': graph.number_of_nodes(),
              'Ground set(Pruned)':len(pruned_universe), 'Queries(Unpruned)': queries_unpruned,'Time(Unpruned)':time_unpruned,
              'Time(Pruned)': time_pruned,
              'Queries(Pruned)': queries_pruned, 'Pruned Ground set(%)': round(Pg,4)*100,
              'Ratio(%)':round(ratio,4)*100, 'Queries(%)': round(queries_pruned/queries_unpruned,4)*100,
              'TimeRatio': time_pruned/time_unpruned

              }

   
    df = pd.DataFrame(df,index=[0])
    save_to_pickle(df,save_file_path)
    print(df)

    ###################################################################################################


    # pruned_ground_ratio=len(pruned_universe)/graph.number_of_nodes()
    # print('Ground set (Ratio):',pruned_ground_ratio)

    # # Subgraph
    # # subgraph = make_subgraph(graph,pruned_universe)

    # # Pv=1-subgraph.number_of_nodes()/graph.number_of_nodes()
    # # Pe=1-subgraph.number_of_edges()/graph.number_of_edges()
    # # print('Pv:',Pv)
    # # print('Pe:',Pe)
    # df=defaultdict(list)
    # for budget in budgets:
    #     solution_subgraph,queries_subgraph = greedy(graph,budget,pruned_universe)
    #     solution_graph,queries_graph= greedy(graph,budget)
    #     pruned_cover= calculate_cover(graph,solution_subgraph)
    #     whole_cover= calculate_cover(graph,solution_graph)

    #     df['Dataset'].append(args.dataset)
    #     df['Size of Ground set'].append(graph.number_of_nodes())
    #     df['Size of Pruned Ground set'].append(len(pruned_universe))
    #     df['Budget'].append(budget)
    #     # df['Solution'].append(solution_subgraph)
    #     df['Objective Value(Pruned)'].append(pruned_cover)
    #     df['Objective Value (Ratio)'].append(pruned_cover/whole_cover)
    #     df['Queries(Ratio)'].append(queries_subgraph/queries_graph)

    # df=pd.DataFrame(df)

    # print(df)

    # save_folder='data/Blimes'
    # file_path=os.path.join(save_folder,args.dataset)
    # os.makedirs(save_folder,exist_ok=True)
    # save_to_pickle(df,file_path)

    




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument( "--dataset", type=str, default='Facebook',required=True, help="Name of the dataset to be used (default: 'Facebook')" )
    parser.add_argument( "--r", type=float, default=8, help="r" )
    parser.add_argument( "--c", type=float, default=8, help="c" )
    # parser.add_argument( "--budgets", nargs='+', type=int, help="Budgets" )
    parser.add_argument("--budget", type=int,required=True,default=10, help="Budgets")

    args = parser.parse_args()

    SS(dataset=args.dataset,r=args.r,c=args.c,budgets=args.budget)
