from argparse import ArgumentParser
from utils import *
import pandas as pd
from collections import defaultdict

from greedy import greedy
import numpy as np
import random
import heapq
import os


def SS(dataset,r,c,budgets):

    file_path=f'../../data/test/{dataset}'
    graph=load_from_pickle(file_path)

    pruned_universe=set()

    universe=list(graph.nodes())
    n=graph.number_of_nodes()
    while len(universe)> r*np.log2(n):
        U=random.sample(universe,int(r*np.log2(n)))
        for node in U:
            universe.remove(node)
        U=set(U)
        pruned_universe=pruned_universe.union(U)

        lst=[]

        universe_gain=calculate_cover(graph,universe)

        for v in universe:

            w=float('inf')
            
            for u in graph.neighbors(v):
                
                if u in U:
                    universe_copy=universe.copy()
                    universe_copy.append(u)
                    
                    local_gain=calculate_cover(graph,[u,v])-calculate_cover(graph,[v])
                    # print(local_gain)

                    global_gain=calculate_cover(graph,universe_copy)-universe_gain
                    w=min(w,local_gain-global_gain)

            lst.append((w,v))

        remove_nodes=heapq.nsmallest(int((1-1/np.sqrt(c))*len(universe)), lst)
        # print(remove_nodes)
        for w,node in remove_nodes:
            # if w>0:
            #     print(w)
            universe.remove(node)

        

    pruned_universe=pruned_universe.union(set(universe))


    pruned_ground_ratio=len(pruned_universe)/graph.number_of_nodes()
    print('Ratio:',pruned_ground_ratio)

    # Subgraph
    subgraph = make_subgraph(graph,pruned_universe)

    Pv=1-subgraph.number_of_nodes()/graph.number_of_nodes()
    Pe=1-subgraph.number_of_edges()/graph.number_of_edges()
    print('Pv:',Pv)
    print('Pe:',Pe)
    df=defaultdict(list)
    for budget in budgets:
        solution_subgraph = greedy(subgraph,budget,pruned_universe)
        cover= calculate_cover(graph,solution_subgraph)

        df['Dataset'].append(args.dataset)
        df['Dataset Path'].append(file_path)
        df['Budget'].append(budget)
        df['Solution'].append(solution_subgraph)
        df['Objective Value'].append(cover)
        df['Objective Value (Ratio)'].append(cover/graph.number_of_nodes())

    df=pd.DataFrame(df)

    print(df)

    save_folder='data/Blimes'
    file_path=os.path.join(save_folder,args.dataset)
    os.makedirs(save_folder,exist_ok=True)
    save_to_pickle(df,file_path)

    




if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        default='Facebook',
        help="Name of the dataset to be used (default: 'Facebook')"
    )

    parser.add_argument(
        "--r",
        type=float,
        default=8,
        help="r"
    )

    parser.add_argument(
        "--c",
        type=float,
        default=8,
        help="c"
    )

    # parser.add_argument(
    #     "--budget",
    #     type=int,
    #     default=20,
    #     help="Budget"
    # )

    parser.add_argument(
        "--budgets",
        nargs='+',
        type=int,
        help="Budgets"
    )


    args = parser.parse_args()

    SS(dataset=args.dataset,r=args.r,c=args.c,budgets=args.budgets)
