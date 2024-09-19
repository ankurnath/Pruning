from utils import *
from greedy import gain_adjustment,get_gains,calculate_spread
from helper_functions import *

from knapsack_greedy import knapsack_greedy


def knapsack_quickfilter(dataset,budget,delta,cost_model,num_rr):


    load_graph_file_path=f'../../data/snap_dataset/{dataset}.txt'

    graph = load_graph(load_graph_file_path)

    node_weights = generate_node_weights(graph=graph,cost_model=cost_model)
    
    start = time.time()
    
    gains,node_rr_set,RR = get_gains(graph,num_rr)
    print(len(RR))
    curr_obj = 0

    pruned_universe=[]
    covered_rr_set = set ()
    for node in graph.nodes():

        if gains[node]/node_weights[node] >= delta/budget*curr_obj:
            curr_obj+=gains[node]
            pruned_universe.append(node)

            # gains adjustment
            gain_adjustment(gains=gains,node_rr_set=node_rr_set,
                            RR=RR,selected_element=node,covered_rr_set=covered_rr_set)
    
    end= time.time()

    time_to_prune = end-start

    
    



    ##################################################################

    Pg=len(pruned_universe)/graph.number_of_nodes()
    print('Size of Pruned Ground Set, |Upruned|:', len(pruned_universe))
    start = time.time()


    solution_unpruned,queries_unpruned = knapsack_greedy (graph=graph,ground_set =None, num_rr=num_rr,budget = budget, node_weights = node_weights)
    objective_unpruned = calculate_spread(graph=graph,solution=solution_unpruned )

    end = time.time()
    time_unpruned = round(end-start,4)
    print('Elapsed time (unpruned):',round(time_unpruned,4))

    start = time.time()

    solution_pruned,queries_pruned = knapsack_greedy (graph=graph,
                                                      ground_set = pruned_universe, 
                                                      num_rr=num_rr,budget = budget, 
                                                      node_weights = node_weights)
    objective_pruned = calculate_spread(graph=graph,solution=solution_pruned )

    
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

    ##################################################################################################
      


        
        

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default='DBLP', help="Name of the dataset to be used (default: 'Facebook')")
    parser.add_argument("--budget", type=int,default=100, help="Budget")
    parser.add_argument("--delta", type=float, default=0.1, help="Delta")
    parser.add_argument("--cost_model",type= str, default= 'degree', help = 'model of node weights')
    parser.add_argument("--num_rr", type=int, default= 100000  , help="Number of RR sets")
    


    args = parser.parse_args()


    dataset = args.dataset
    
    cost_model = args.cost_model
    budget = args.budget
    delta = args.delta
    num_rr = args.num_rr


    sprint(dataset)
    sprint(budget)
    sprint(cost_model)
    sprint(delta)


    knapsack_quickfilter(dataset=dataset,budget = budget,
                         delta = delta,cost_model = cost_model
                         ,num_rr=num_rr)

  
