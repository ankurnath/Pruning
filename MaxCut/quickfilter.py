from utils import *
from greedy import greedy,gain_adjustment,get_gains


def quickfilter(dataset,budget,delta=0.1):

    sprint(dataset)
    sprint(budget)
    sprint(delta)
    

    load_graph_file_path=f'../../data/snap_dataset/{dataset}.txt'
    graph = load_graph(load_graph_file_path)
    
    start = time.time()
    gains = get_gains(graph,ground_set=None)
    curr_obj = 0
    pruned_universe = []
    uncovered = defaultdict(lambda: True)
    for node in tqdm(graph.nodes()):

        if gains[node]>=delta/budget*curr_obj:
            curr_obj+=gains[node]
            pruned_universe.append(node)
            gain_adjustment(graph,gains,node,uncovered)
            
    end= time.time()

    time_to_prune = end-start

    print('time elapsed to pruned',time_to_prune)
    
    
    ##################################################################

    Pg=len(pruned_universe)/graph.number_of_nodes()
    start = time.time()
    objective_unpruned,queries_unpruned,solution_unpruned= greedy(graph,budget)
    end = time.time()
    time_unpruned = round(end-start,4)
    print('Elapsed time (unpruned):',round(time_unpruned,4))

    start = time.time()
    objective_pruned,queries_pruned,solution_pruned = greedy(graph=graph,budget=budget,ground_set=pruned_universe)
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
    print('Queries:',round(queries_pruned/queries_unpruned,4)*100)


    save_folder = f'data/{dataset}'
    os.makedirs(save_folder,exist_ok=True)
    save_file_path = os.path.join(save_folder,'Quickfilter')

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
    parser.add_argument("--dataset", type=str, default='Facebook',required=True, help="Name of the dataset to be used (default: 'Facebook')")
    parser.add_argument("--budget", type=int,required=True,default=10, help="Budgets")
    parser.add_argument("--delta", type=float, default=0.1,required=True, help="Delta")
    args = parser.parse_args()
    quickfilter(dataset=args.dataset,budget=args.budget)
