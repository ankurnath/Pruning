from argparse import ArgumentParser
from utils import *
import pandas as pd
from collections import defaultdict

from greedy import greedy,gain_adjustment,get_gains



def quickfilter(dataset,budgets,delta=0.1):
    # load_graph_file_path=f'../../data/test/{dataset}'
    # graph=load_from_pickle(load_graph_file_path)

    load_graph_file_path=f'../../data/snap_dataset/{dataset}.txt'
    graph=nx.read_edgelist(f'../../data/snap_dataset/{dataset}.txt', create_using=nx.Graph(), nodetype=int)

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
                

        
        # Pg=1-len(pruned_universe)/graph.number_of_nodes()
        Pg=len(pruned_universe)/graph.number_of_nodes()

        print('Pruned Universe:',len(pruned_universe))
        
        
        # Subgraph 
        subgraph =make_subgraph(graph,pruned_universe)

        
        Pv=1-subgraph.number_of_nodes()/graph.number_of_nodes()
        Pe=1-subgraph.number_of_edges()/graph.number_of_edges()
        
        solution_subgraph,queries_subgraph = greedy(graph,budget,pruned_universe)

        greedy_solution,queries_graph= greedy(graph,budget)
        print()

        coverage= calculate_cover(graph,solution_subgraph)
        df['Dataset'].append(args.dataset)
        df['Budget'].append(budget)
        df['Size of Ground set'].append(graph.number_of_nodes())
        df['Size of Pruned Ground set'].append(len(pruned_universe))
        # df['Budget'].append(budget)
        # df['Pv'].append(Pv)
        # df['Pe'].append(Pe)
        # df['Pg'].append(Pg)
        df['Objective Value(Pruned)'].append(coverage)
        
        df['Objective Value (Ratio)'].append(coverage/calculate_cover(graph,greedy_solution))
        df['Queries(Ratio)'].append(queries_subgraph/queries_graph)
        # df['Solution'].append(solution_subgraph)
        # df['Objective Value (Ratio)'].append(coverage/graph.number_of_nodes())
        

    df['delta']=[delta]*len(df['Budget'])
    # print(df)
    df=pd.DataFrame(df)

    
    # try:
    #     df['Ratio']=df['Objective Value']/load_from_pickle(f'data/{args.dataset}/Greedy')['Objective Value']
    # except:
    #     raise ValueError('Greedy value is not found.')
    print(df)



    save_folder=f'data/{args.dataset}'
    file_path=os.path.join(save_folder,'QuickFilter')
    os.makedirs(save_folder,exist_ok=True)
    save_to_pickle(df,file_path) 

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default='Facebook', help="Name of the dataset to be used (default: 'Facebook')")
    parser.add_argument("--budgets", nargs='+', type=int, help="Budgets")
    parser.add_argument("--delta", type=float, default=0.1, help="Delta")

    # parser.add_argument(
    #     "--dataset",
    #     type=str,
    #     default='Facebook',
    #     help="Name of the dataset to be used (default: 'Facebook')"
    # )
    # parser.add_argument(
    #     "--budgets",
    #     nargs='+',
    #     type=int,
    #     help="Budgets"
    # )
    # parser.add_argument(
    #     "--delta",
    #     type=float,
    #     default=0.1,
    #     help="Delta"
    # )


    args = parser.parse_args()


    # for budget in args.budgets:

    quickfilter(dataset=args.dataset,budgets=args.budgets)
