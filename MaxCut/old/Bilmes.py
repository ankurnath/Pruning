from argparse import ArgumentParser
from utils import load_from_pickle,make_subgraph,calculate_cut
import pandas as pd
from collections import defaultdict

from greedy import greedy
import numpy as np
import random
import heapq


def SS(dataset,r,c,budget):
    graph=load_from_pickle(f'../../data/test/{dataset}')

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

        universe_gain=calculate_cut(graph,universe)

        for v in universe:

            w=float('inf')
            
            for u in graph.neighbors(v):
                
                if u in U:
                    universe_copy=universe.copy()
                    universe_copy.append(u)
                    
                    local_gain=calculate_cut(graph,[u,v])-calculate_cut(graph,[v])
                    # print(local_gain)

                    global_gain=calculate_cut(graph,universe_copy)-universe_gain
                    w=min(w,local_gain-global_gain)

            lst.append((w,v))

        remove_nodes=heapq.nsmallest(int((1-1/np.sqrt(c))*len(universe)), lst)
        # print(remove_nodes)
        for w,node in remove_nodes:
            # if w>0:
            #     print(w)
            universe.remove(node)

        

    pruned_universe=pruned_universe.union(set(universe))

    print('Ratio:',len(pruned_universe)/graph.number_of_nodes())

    # Subgraph
    subgraph =make_subgraph(graph,pruned_universe)
    print('Pv:',1-subgraph.number_of_nodes()/graph.number_of_nodes())
    print('Pe:',1-subgraph.number_of_edges()/graph.number_of_edges())

    solution_subgraph = greedy(subgraph,budget)

    coverage= calculate_cut(graph,solution_subgraph)

    print('Cut',coverage/graph.number_of_edges())




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

    parser.add_argument(
        "--budget",
        type=int,
        default=20,
        help="Budget"
    )


    args = parser.parse_args()

    SS(dataset=args.dataset,r=args.r,c=args.c,budget=args.budget)
