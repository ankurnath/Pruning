from utils import *
from greedy import get_gains
# from DLA_numba import DLA
from knapsack_greedy import knapsack_greedy,calculate_spread


def top_k(dataset,budget,cost_model):
    load_graph_file_path=f'../../data/snap_dataset/{dataset}.txt'

    graph = load_graph(load_graph_file_path)

    node_weights = generate_node_weights(graph=graph,cost_model=cost_model)


    size = load_from_pickle(f'data/{dataset}/knapsack_multi/Quickfilter_degree')['Ground set Single (Pruned)'].iloc[-1]
    gains,node_rr_set,RR = get_gains(graph,num_rr=100000)
    objective_unpruned = load_from_pickle(f'data/{dataset}/knapsack_multi/Quickfilter_{cost_model}')['Objective Value(Unpruned)'].iloc[-1]
    sprint(size)

    
    density_gain = {node: gains[node]/node_weights[node] for node in gains}
    pruned_universe = [key for key, _ in sorted(density_gain.items(), key=lambda item: item[1], reverse=True)[:size]]
    Pg=len(pruned_universe)/graph.number_of_nodes()
    sprint(Pg)

    
    solution_pruned,_=   knapsack_greedy (     graph=graph,
                                          budget=budget,
                                          node_weights=node_weights,
                                          ground_set=pruned_universe,
                                          gains=gains.copy(),
                                          node_rr_set=node_rr_set,
                                          RR=RR,
                                          
                                          )
    
    objective_pruned = calculate_spread(graph=graph,solution=solution_pruned )
    ratio = objective_pruned/objective_unpruned


    save_folder = f'data/{dataset}/knapsack'
    os.makedirs(save_folder,exist_ok=True)
    save_file_path = os.path.join(save_folder,f'top_K_{cost_model}')

    df ={     'Dataset':dataset,
              'Budget':budget,
            #   'Objective Value(Unpruned)':objective_unpruned,
            #   'Objective Value(Pruned)':objective_pruned ,
            #   'Ground Set': graph.number_of_nodes(),
            #   'Ground set(Pruned)':len(pruned_universe), 
            #   'Queries(Unpruned)': queries_unpruned,'Time(Unpruned)':time_unpruned,
            #   'Time(Pruned)': time_pruned,
            #   'Queries(Pruned)': queries_pruned, 
            #   'Pruned Ground set(%)': round(Pg,4)*100,
              'Ratio(%)':round(ratio,4)*100, 
            #   'Queries(%)': round(queries_pruned/queries_unpruned,4)*100,
            #   'TimeRatio': time_pruned/time_unpruned,
              

              }

   
    df = pd.DataFrame(df,index=[0])
    save_to_pickle(df,save_file_path)
    print(df)




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument( "--dataset", type=str, default='Facebook', help="Name of the dataset to be used (default: 'Facebook')" )
    parser.add_argument("--budget", type=int,default=100, help="Budget")
    parser.add_argument("--cost_model",type= str, default= 'degree', help = 'model of node weights')
  
    args = parser.parse_args()

    dataset = args.dataset
    
    cost_model = args.cost_model
    budget = args.budget

    top_k(dataset,budget,cost_model)