from argparse import ArgumentParser
from utils import load_from_pickle,make_subgraph,calculate_cover
import pandas as pd
from collections import defaultdict

from greedy import greedy



def quickfilter(dataset,budget,delta=0.1):
    graph=load_from_pickle(f'../../data/test/{dataset}')

    # pruning stage

    gains={node:graph.degree(node) for node in graph.nodes()}

    curr_obj=0

    pruned_universe=[]
    uncovered=defaultdict(lambda: True)
    for node in graph.nodes():

        if gains[node]>=delta/budget*curr_obj:
            curr_obj+=gains[node]
            pruned_universe.append(node)

            # gains adjustment
            if uncovered[node]:
                gains[node]-=1
                uncovered[node]=False
                for neighbour in graph.neighbors(node):
                    if gains[neighbour]>0:
                        gains[neighbour]-=1

            for neighbour in graph.neighbors(node):
                if uncovered[neighbour]:
                    uncovered[neighbour]=False
                    
                    if  neighbour in gains:
                        gains[neighbour]-=1
                    
                    for neighbour_of_neighbour in graph.neighbors(neighbour):

                        gains[neighbour_of_neighbour]-=1

    print(len(pruned_universe))
    print('Ratio of Ground Set:',len(pruned_universe)/graph.number_of_nodes())
    
    # Subgraph
    subgraph =make_subgraph(graph,pruned_universe)
    print('Pv:',1-subgraph.number_of_nodes()/graph.number_of_nodes())
    print('Pe:',1-subgraph.number_of_edges()/graph.number_of_edges())

    solution_subgraph = greedy(subgraph,budget)

    coverage= calculate_cover(graph,solution_subgraph)

    print('Coverage:',coverage/graph.number_of_nodes())





if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        default='Facebook',
        help="Name of the dataset to be used (default: 'Facebook')"
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=20,
        help="Budget"
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.1,
        help="Delta"
    )


    args = parser.parse_args()

    quickfilter(dataset=args.dataset,budget=args.budget)
