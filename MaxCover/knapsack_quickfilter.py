from argparse import ArgumentParser
from utils import *
import pandas as pd
from collections import defaultdict

from greedy import gain_adjustment,get_gains
from budgeted_greedy import modified_greedy


def quickfilter(dataset,budgets,delta=0.1,node_weights=None):
    load_graph_file_path=f'../../data/test/{dataset}'
    graph=load_from_pickle(load_graph_file_path)

    # load_graph_file_path=f'../../data/snap_dataset/{dataset}.txt'
    # graph=nx.read_edgelist(f'../../data/snap_dataset/{dataset}.txt', create_using=nx.Graph(), nodetype=int)

    # pruning stage

    # gains={node:graph.degree(node) for node in graph.nodes()}
    df=defaultdict(list)

    if node_weights is None:
        node_weights = {node:1 for node in graph.nodes()}

    
    for budget in budgets:
        gains=get_gains(graph,ground_set=None)

        

        curr_obj=0

        pruned_universe=[]
        uncovered=defaultdict(lambda: True)
        for node in graph.nodes():

            if gains[node]/node_weights[node]>=delta/budget*curr_obj:
                curr_obj+=gains[node]
                pruned_universe.append(node)

                # gains adjustment
                gain_adjustment(graph,gains,node,uncovered)
                

        
        Pg=1-len(pruned_universe)/graph.number_of_nodes()

        print('Pruned Universe:',len(pruned_universe))
        
        
        # Subgraph 
        subgraph =make_subgraph(graph,pruned_universe)

        
        Pv=1-subgraph.number_of_nodes()/graph.number_of_nodes()
        Pe=1-subgraph.number_of_edges()/graph.number_of_edges()
        
        solution_subgraph,_ = modified_greedy(graph=graph,budget=budget,ground_set=pruned_universe,node_weights=node_weights)
        print(solution_subgraph)
        print([node_weights[node] for node in solution_subgraph])

        greedy_solution,_ = modified_greedy(graph=graph,budget=budget,node_weights=node_weights)
        print(greedy_solution)
        print([node_weights[node] for node in greedy_solution])
        print()

        coverage= calculate_cover(graph,solution_subgraph)

        df['Budget'].append(budget)
        df['Pv'].append(Pv)
        df['Pe'].append(Pe)
        df['Pg'].append(Pg)
        df['Objective Value'].append(coverage)
        df['Ratio'].append(calculate_cover(graph,greedy_solution)/coverage)
        # df['Solution'].append(solution_subgraph)
        # df['Objective Value (Ratio)'].append(coverage/graph.number_of_nodes())
        

    df['delta']=[delta]*len(df['Budget'])

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
    parser.add_argument("--cost_model",type= str, default= 'random', help = 'model of node weights')
    


    args = parser.parse_args()

    node_weights = load_from_pickle(f'../../data/test/{args.dataset}_weights_{args.cost_model}')
    # for budget in args.budgets:

    quickfilter(dataset=args.dataset,budgets=args.budgets,node_weights=node_weights)
