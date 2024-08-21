from utils import *
from greedy import gain_adjustment,get_gains
from knapsack_greedy import knapsack_greedy
from knapsack_numba_greedy import knapsack_numba_greedy
from IP_solver import gurobi_solver

def knapsack_quickfilter(dataset,budget,delta,cost_model):


    load_graph_file_path=f'../../data/snap_dataset/{dataset}.txt'

    graph = load_graph(load_graph_file_path)

    node_weights = generate_node_weights(graph=graph,cost_model=cost_model)
    
    start = time.time()
    # for budget in budgets:
    gains = get_gains(graph,ground_set=None)
    curr_obj = 0

    pruned_universe=[]
    uncovered=defaultdict(lambda: True)
    for node in graph.nodes():

        if gains[node]/node_weights[node] >= delta/budget*curr_obj:
            curr_obj+=gains[node]
            pruned_universe.append(node)

            # gains adjustment
            gain_adjustment(graph,gains,node,uncovered)
    
    end= time.time()

    time_to_prune = end-start

    print('time elapsed to pruned',time_to_prune)


        
    subgraph = make_subgraph(graph,pruned_universe)
    # solution_unpruned,objval_unpruned= gurobi_solver(graph=graph,budget=budget,node_weights=node_weights)
    # solution_pruned,objval_pruned= gurobi_solver(graph=subgraph,budget=budget,node_weights=node_weights)



    ##################################################################

    Pg=len(pruned_universe)/graph.number_of_nodes()
    start = time.time()
    # objective_unpruned,queries_unpruned,solution_unpruned= knapsack_numba_greedy(graph=graph,budget=budget,
    #                                                                              node_weights=node_weights)
    
    objective_unpruned,queries_unpruned,solution_unpruned = gurobi_solver(graph=graph,budget=budget,
                                                                          node_weights=node_weights)
    end = time.time()
    time_unpruned = round(end-start,4)
    print('Elapsed time (unpruned):',round(time_unpruned,4))

    start = time.time()
    # objective_pruned,queries_pruned,solution_pruned = knapsack_numba_greedy(graph=graph,budget=budget,
    #                                                                         node_weights=node_weights,
    #                                                                         ground_set=pruned_universe)
    
    objective_pruned,queries_pruned,solution_pruned = gurobi_solver(graph=subgraph,budget=budget,node_weights=node_weights)
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
    print('Queries:',round(queries_pruned/queries_unpruned,4)*100)


    save_folder = f'data/{dataset}/knapsack'
    os.makedirs(save_folder,exist_ok=True)
    save_file_path = os.path.join(save_folder,f'Quickfilter_{cost_model}')

    df ={     'Dataset':dataset,'Budget':budget,'Delta':delta,'Objective Value(Unpruned)':objective_unpruned,
              'Objective Value(Pruned)':objective_pruned ,'Ground Set': graph.number_of_nodes(),
              'Ground set(Pruned)':len(pruned_universe), 'Queries(Unpruned)': queries_unpruned,'Time(Unpruned)':time_unpruned,
              'Time(Pruned)': time_pruned,
              'Queries(Pruned)': queries_pruned, 'Pruned Ground set(%)': round(Pg,4)*100,
              'Ratio(%)':round(ratio,4)*100, 'Queries(%)': round(queries_pruned/queries_unpruned,4)*100,
              'TimeRatio': time_pruned/time_unpruned,
              'TimeToPrune':time_to_prune

              }

   
    df = pd.DataFrame(df,index=[0])
    save_to_pickle(df,save_file_path)
    print(df)

    ###################################################################################################
      


        
        

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default='DBLP', help="Name of the dataset to be used (default: 'Facebook')")
    parser.add_argument("--budget", type=int,default=5, help="Budget")
    parser.add_argument("--delta", type=float, default=0.1, help="Delta")
    parser.add_argument("--cost_model",type= str, default= 'degree', help = 'model of node weights')
    


    args = parser.parse_args()


    dataset = args.dataset
    
    cost_model = args.cost_model
    budget = args.budget
    delta = args.delta
    cost_model = args.cost_model


    knapsack_quickfilter(dataset=dataset,budget = budget,delta = delta,cost_model = cost_model)

  
