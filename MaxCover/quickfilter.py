from argparse import ArgumentParser
from utils import *
import pandas as pd
from collections import defaultdict

from greedy import greedy,gain_adjustment,get_gains



def quickfilter(dataset,budgets,delta=0.1):
    load_graph_file_path=f'../../data/test/{dataset}'
    graph=load_from_pickle(load_graph_file_path)

    # pruning stage

    # gains={node:graph.degree(node) for node in graph.nodes()}
    df=defaultdict(list)
    for budget in budgets:
        gains=get_gains(graph,ground_set=None)

        curr_obj=0

        pruned_universe=[]
        uncovered=defaultdict(lambda: True)
        for node in graph.nodes():

            if gains[node]>=delta/budget*curr_obj:
                curr_obj+=gains[node]
                pruned_universe.append(node)

                # gains adjustment
                gain_adjustment(graph,gains,node,uncovered)
                

        # print(len(pruned_universe))
        ground_set_ratio=len(pruned_universe)/graph.number_of_nodes()
        print('Ratio of Ground Set:',ground_set_ratio)
        
        # Subgraph 
        subgraph =make_subgraph(graph,pruned_universe)

        Pv=1-subgraph.number_of_nodes()/graph.number_of_nodes()
        Pe=1-subgraph.number_of_edges()/graph.number_of_edges()
        print('Pv:',Pv)
        print('Pe:',Pe)
        # print('Pv:',1-subgraph.number_of_nodes()/graph.number_of_nodes())
        # print('Pe:',1-subgraph.number_of_edges()/graph.number_of_edges())

        # solution_subgraph = greedy(subgraph,budget)
        solution_subgraph = greedy(subgraph,budget,pruned_universe)

        coverage= calculate_cover(graph,solution_subgraph)

        # print('if whole subgraph is the ground set,Coverage:',coverage/graph.number_of_nodes())

        # solution_subgraph = greedy(subgraph,budget,pruned_universe)

        # coverage= calculate_cover(graph,solution_subgraph)

        df['Dataset'].append(args.dataset)
        df['Pruned Ground Set (Ratio)'].append(len(pruned_universe)/graph.number_of_nodes())
        df['Pv'].append(Pv)
        df['Pe'].append(Pe)
        df['Budget'].append(budget)
        df['Dataset Path'].append(load_graph_file_path)
        
        
        df['Solution'].append(solution_subgraph)
        df['Objective Value'].append(coverage)
        df['Objective Value (Ratio)'].append(coverage/graph.number_of_nodes())
        # print('if pruned universeis the ground set,Coverage:',coverage/graph.number_of_nodes())

    df=pd.DataFrame(df)

    print(df)

    save_folder='data/quickfilter'
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
        "--budgets",
        nargs='+',
        type=int,
        help="Budgets"
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.1,
        help="Delta"
    )


    args = parser.parse_args()


    # for budget in args.budgets:

    quickfilter(dataset=args.dataset,budgets=args.budgets)
