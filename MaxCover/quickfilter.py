from argparse import ArgumentParser
from utils import *
import pandas as pd
from collections import defaultdict

from greedy import greedy,gain_adjustment,get_gains




def quickfilter(dataset,budget,delta=0.1):

    sprint(dataset)
    sprint(budget)
    sprint(delta)
    

    load_graph_file_path=f'../../data/snap_dataset/{dataset}.txt'
    graph=nx.read_edgelist(f'../../data/snap_dataset/{dataset}.txt', create_using=nx.Graph(), nodetype=int)

    
    
    
    gains=get_gains(graph,ground_set=None)

    curr_obj=0

    pruned_universe=[]
    uncovered=defaultdict(lambda: True)
    for node in graph.nodes():

        if gains[node]>=delta/budget*curr_obj:
            curr_obj+=gains[node]
            pruned_universe.append(node)
            gain_adjustment(graph,gains,node,uncovered)
            

    
    
    Pg=len(pruned_universe)/graph.number_of_nodes()

    start = time.time()
    solution_unpruned,queries_unpruned= greedy(graph,budget)
    end = time.time()
    solution_pruned,queries_pruned = greedy(graph=graph,budget=budget,ground_set=pruned_universe)

    
    
    objective_unpruned = calculate_cover(graph,solution_unpruned)
    objective_pruned = calculate_cover(graph,solution_pruned)
    
    ratio = objective_pruned/objective_unpruned


    print('Performance of teacher Model')
    print('Size Constraint,k:',budget)
    print('Size of Ground Set,|U|:',graph.number_of_nodes())
    print('Size of Pruned Ground Set, |Upruned|:', len(solution_unpruned))
    print('Pg(%):', round(Pg,4)*100)
    print('Ratio:',round(ratio,4)*100)
    print('Queries:',round(queries_pruned/queries_unpruned,4)*100)

    save_file_path = 'data/Quickfilter'

    if os.path.exists(save_file_path):
        df = load_from_pickle(save_file_path)

    else:
        column_names = ['Dataset','Budget','Delta','Objective Value(Unpruned)',
                        'Objective Value(Pruned)','Ground Set','Ground set(Pruned)',
                        'Queries(Unpruned)','Queries(Pruned)',
                        'Pruned Ground set(%)','Ratio(%)','Queries(%)']
        df = pd.DataFrame(columns=column_names)

    new_row ={'Dataset':dataset,'Budget':budget,'Delta':delta,'Objective Value(Unpruned)':objective_unpruned,
              'Objective Value(Pruned)':objective_pruned ,'Ground Set': graph.number_of_nodes(),
              'Ground set(Pruned)':len(pruned_universe), 'Queries(Unpruned)': queries_unpruned,
              'Queries(Pruned)': queries_pruned, 'Pruned Ground set(%)': round(Pg,4)*100,
              'Ratio(%)':round(ratio,4)*100, 'Queries(%)': round(queries_pruned/queries_unpruned,4)*100
              }

    df = df._append(new_row, ignore_index=True)

    # df=defaultdict(list)
    # df['Dataset'].append(args.dataset)
    # df['Budget'].append(budget)
    # df['Size of Ground set'].append(graph.number_of_nodes())
    # df['Size of Pruned Ground set'].append(len(pruned_universe))
    # df['Objective Value(Pruned)'].append(coverage)
    
    # df['Objective Value (Ratio)'].append(coverage/calculate_cover(graph,greedy_solution))
    # df['Queries(Ratio)'].append(queries_subgraph/queries_graph)
        
        

    # df['delta']=[delta]*len(df['Budget'])
 
    # df=pd.DataFrame(df)

    

    # print(df)



    # save_folder=f'data/{args.dataset}'
    # file_path=os.path.join(save_folder,'QuickFilter')
    # os.makedirs(save_folder,exist_ok=True)
    save_to_pickle(df,save_file_path)
    print(df)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default='Facebook', help="Name of the dataset to be used (default: 'Facebook')")
    parser.add_argument("--budget", type=int,default=10, help="Budgets")
    parser.add_argument("--delta", type=float, default=0.1, help="Delta")
    args = parser.parse_args()
    quickfilter(dataset=args.dataset,budget=args.budget)
