from utils import *
from greedy import get_gains
from knapsack_numba_greedy import knapsack_numba_greedy
from helper_functions import *

def top_k(dataset,budget,cost_model):

    load_graph_file_path=f'../../data/snap_dataset/{dataset}.txt'

    graph = load_graph(load_graph_file_path)

    node_weights = generate_node_weights(graph=graph,cost_model=cost_model)


    size = load_from_pickle(f'data/{dataset}/knapsack_multi/Quickfilter_aistats')['Ground set Single (Pruned)'].iloc[-1]
    
    objective_unpruned = load_from_pickle(f'data/{dataset}/knapsack_multi/Quickfilter_aistats')['Objective Value(Unpruned)'].iloc[-1]
    sprint(size)

    gains = get_gains(graph,ground_set=None)

    max_weight_node=max(node_weights, key=node_weights.get)

    sprint(node_weights[max_weight_node])

    density_gain = {node: gains[node]/node_weights[node] for node in gains}
    
    # max_density_gain_node=max(density_gain, key=density_gain.get)
    # sprint(density_gain[max_density_gain_node])
    # density_gain = {'a': 5, 'b': 3, 'c': 8, 'd': 1}

    # sorted_density_gain = sorted(density_gain.items(), key=lambda item: item[1], reverse=True)
    # sprint(sorted_density_gain)


    # Assuming `density_gain` is a dictionary where keys are nodes and values are the density gains
    # Sort the dictionary by values (density gains) in descending order and select the top 100 elements
    # Get the top `size` keys based on their density gain values
    pruned_universe = [key for key, item in sorted(density_gain.items(), key=lambda item: item[1], reverse=True) if node_weights[key] <= budget][:size]

    # sprint(sorted(density_gain.items(), key=lambda item: item[1], reverse=True)[:10])
    # pruned_universe = [key for key, item in sorted(density_gain.items(), key=lambda item: item[1], reverse=True)][:size]
    # print([node_weights[node] for node in pruned_universe])
    # print([density_gain[node] for node in pruned_universe])
    # # top_k_elements = dict(sorted(density_gain.items(), key=lambda item: item[1], reverse=True)[:size])
    Pg=len(pruned_universe)/graph.number_of_nodes()
    # start = time.time()
    # objective_unpruned,queries_unpruned,solution_unpruned= knapsack_numba_greedy(graph=graph,budget=budget,
    #                                                                              node_weights=node_weights)
    # end = time.time()
    # time_unpruned = round(end-start,4)
    # print('Elapsed time (unpruned):',round(time_unpruned,4))

    start = time.time()
    objective_pruned,queries_pruned,solution_pruned = knapsack_numba_greedy(graph=graph,budget=budget,
                                                                            node_weights=node_weights,
                                                                            ground_set=pruned_universe)
    end = time.time()
    time_pruned = round(end-start,4)
    print('Elapsed time (pruned):',time_pruned)
    ratio = objective_pruned/objective_unpruned


    print('Performance of QuickFilter')
    print('Size Constraint,k:',budget)
    print('Size of Ground Set,|U|:',graph.number_of_nodes())
    print('Size of Pruned Ground Set, |Upruned|:', len(pruned_universe))
    print('Pg(%):', round(Pg,4)*100)
    print('Ratio:',round(ratio,4)*100)

    # print(queries_pruned)
    # print(queries_unpruned)
    # print('Queries:',round(queries_pruned/queries_unpruned,4)*100)


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
              'Time(Pruned)': time_pruned,
              'Queries(Pruned)': queries_pruned, 
              'Pruned Ground set(%)': round(Pg,4)*100,
              'Ratio(%)':round(ratio,4)*100, 
            #   'Queries(%)': round(queries_pruned/queries_unpruned,4)*100,
            #   'TimeRatio': time_pruned/time_unpruned,
              

              }

   
    df = pd.DataFrame(df,index=[0])
    save_to_pickle(df,save_file_path)
    print(df)

    # print(top_k_keys)
    # Now `top_100_elements` contains the top 100 nodes and their corresponding density gains







if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument( "--dataset", type=str, default='Facebook', help="Name of the dataset to be used (default: 'Facebook')" )
    parser.add_argument("--budget", type=int,default=100, help="Budget")
    parser.add_argument("--cost_model",type= str, default= 'aistats', help = 'model of node weights')
  
    args = parser.parse_args()

    dataset = args.dataset
    
    cost_model = args.cost_model
    budget = args.budget

    top_k(dataset,budget,cost_model)