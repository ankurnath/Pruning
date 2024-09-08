from utils import *

from greedy import greedy,gain_adjustment,get_gains

from knapsack_numba_greedy import knapsack_numba_greedy
import matplotlib.pyplot as plt
# from IP_solver import gurobi_solver
# from sample_greedy import run_sampling_multiple_times
from DLA_numba import DLA



def quickfilter_multi(dataset, cost_model , max_budget, min_budget,delta ,eps,args):

    load_graph_file_path=f'../../data/snap_dataset/{dataset}.txt'
    graph = load_graph(load_graph_file_path)
    node_weights = generate_node_weights(graph=graph,cost_model=cost_model)


    start = time.time()
   
    # df = defaultdict(list)
    u_taus = {}
    gains_taus ={}
    uncovered_taus = {}
    
    m = int(np.ceil (np.log(max_budget/min_budget)/np.log(1+eps)))
    print ('m =',m)
    curr_obj_taus = defaultdict(int)
    for i in range(m+1):
        tau = (1+eps)**i * min_budget
        u_taus [i] =set([])
        gains_taus [i] = get_gains(graph,ground_set=None)
        uncovered_taus[i] = defaultdict(lambda: True)
        
    for node in graph.nodes():
        for i in range(m+1):
            tau = (1+eps)**i * min_budget
            if gains_taus[i][node]/node_weights[node]>=(delta/tau)*curr_obj_taus[i]:
                curr_obj_taus[i]+=gains_taus[i][node]
                u_taus [i].add(node)
                # gains adjustment
                gain_adjustment(graph,gains_taus[i],node,uncovered_taus[i])
            

    
    for key in u_taus:
        print(f'key:{key} tau:{int((1+eps)**key * min_budget)} size:{len(u_taus[key])}')


    u = u_taus [0]

    for i in range(1,m+1):
        u = u.union(u_taus[i])

    pruned_universe_multi = list(u)

    end = time.time()

    timetoprune_multi = end-start

    Pg_multi=len(pruned_universe_multi)/graph.number_of_nodes()
    Pg_multi = round(Pg_multi,4)*100
    print("Pg(%):",Pg_multi)
    print('Multi budget Pruned Universe:',len(pruned_universe_multi))
    print("Multi budget Pruned Universe in percentage:",Pg_multi)



    start = time.time()
    gains = get_gains(graph,ground_set=None)
    curr_obj=0
    pruned_universe_single=[]
    uncovered=defaultdict(lambda: True)
    for node in graph.nodes():

        if gains[node]/node_weights[node]>=delta/max_budget*curr_obj:
            curr_obj+=gains[node]
            pruned_universe_single.append(node)

            # gains adjustment
            gain_adjustment(graph,gains,node,uncovered)   
    

    Pg_single = round(len(pruned_universe_single)/graph.number_of_nodes(),4)*100
    print(f'Single budget Size of Pruned universe:{len(pruned_universe_single)}')
    print("Single budget Pruned Universe in percentage:",Pg_single)

    end= time.time()

    timetoprune_single = end - start
    
    


    df = defaultdict(list)


    step = 20
    budgets = list(range(min_budget,max_budget,step)) +[max_budget]
    sprint(budgets)

    for i in budgets:

        print(i)

        start = time.time()

        objective_multi_pruned=DLA(graph=graph,
                                    budget=i,
                                    node_weights=node_weights,
                                    ground_set=pruned_universe_multi,
                                    )

        
        # objective_multi_pruned,queries_multi_pruned =run_sampling_multiple_times(graph=graph,
        #                                                                          budget=i,
        #                                                                          node_weights=node_weights,
        #                                                                          ground_set=pruned_universe_multi,
        #                                                                          num_iterations=5)
        
        end = time.time()

        time_multi_pruned = end -start

        start = time.time()

        objective_single_pruned=DLA(graph=graph,
                                    budget=i,
                                    node_weights=node_weights,
                                    ground_set=pruned_universe_single,
                                    )

        # objective_single_pruned,queries_single_pruned =run_sampling_multiple_times(graph=graph,
        #                                                                          budget=i,
        #                                                                          node_weights=node_weights,
        #                                                                          ground_set=pruned_universe_single,
        #                                                                          num_iterations=5)

        
        end = time.time()
        time_single_pruned = end -start

        start = time.time()

        objective_unpruned= DLA(        graph=graph,
                                        budget=i,
                                        node_weights=node_weights,
                                        ground_set=None,
                                        )


        # objective_unpruned,queries_unpruned =run_sampling_multiple_times(        graph=graph,
        #                                                                          budget=i,
        #                                                                          node_weights=node_weights,
        #                                                                          ground_set=None,
        #                                                                          num_iterations=5)

        
        end = time.time()

        time_unpruned = end- start
        sprint(objective_multi_pruned)
        sprint(objective_single_pruned)
        sprint(objective_unpruned)
        
        df['Dataset'].append(dataset)
        df['Budget'].append(i)
        df['Delta'].append(delta)
        df['Objective Value(Unpruned)'].append(objective_unpruned)
        df['Objective Value Multi(Pruned)'].append(objective_multi_pruned)
        df['Objective Value Single(Pruned)'].append(objective_single_pruned)
        df['Ground Set'].append(graph.number_of_nodes())
        df['Ground set Multi (Pruned)'].append(len(pruned_universe_multi))
        df['Ground set Single (Pruned)'].append(len(pruned_universe_single))
        
        df['Time(Unpruned)'].append(time_unpruned)
        df['Time Multi(Pruned)'].append(time_multi_pruned)
        df['Time Single(Pruned)'].append(time_single_pruned)
        # df['Queries (Unpruned)'].append(queries_unpruned)
        # df['Queries Multi (pruned)'].append(queries_multi_pruned)
        # df['Queries Single (pruned)'].append(queries_single_pruned)

        df['Pruned Ground set Multi(%)'].append(Pg_multi)
        df['Pruned Ground set Single(%)'].append(Pg_single)

        df['Ratio Multi'].append(round(objective_multi_pruned/objective_unpruned,4)*100)
        df['Ratio Single'].append(round(objective_single_pruned/objective_unpruned,4)*100)

        # df['Queries Multi(%)'].append(round(queries_multi_pruned/queries_unpruned,4)*100)
        # df['Queries Single(%)'].append(round(queries_single_pruned/queries_unpruned,4)*100)

        df['TimeRatio(Multi)'].append(time_multi_pruned/time_unpruned)
        df['TimeRatio(Single)'].append(time_single_pruned/time_unpruned)

        df['TimeToPrune(Multi)'].append(timetoprune_multi)
        df['TimeToPrune(Single)'].append(timetoprune_single)




    df = pd.DataFrame(df)

    save_folder = f'data/{dataset}/knapsack_multi'
    os.makedirs(save_folder,exist_ok=True)
    save_file_path = os.path.join(save_folder,f'Quickfilter_{cost_model}')
    save_to_pickle(df,save_file_path)


    print(df[['Budget','Ratio Multi','Ratio Single','Objective Value(Unpruned)']])

        
    fontsize = 20
    plt.plot(budgets, df['Ratio Multi'], linestyle='--', marker='o', markersize=20, color='blue', markeredgecolor='black', alpha=0.7, label=f'Multi-Budget {Pg_multi:.2f}%')
    plt.plot(budgets, df['Ratio Single'], linestyle='--', marker='*', markersize=20, color='red', markeredgecolor='black', alpha=0.7, label=f'Single-Budget {Pg_single:.2f}%')
    
    
    plt.xlabel('Budgets', fontsize=fontsize )
    plt.ylabel('Ratios (%)', fontsize=fontsize)
    plt.title(f'Dataset: {args.dataset} cost_model: {cost_model} Sampled: No Eps: {eps} Delta: {delta} \n'
          f'Max Budget: {max_budget} Min Budget: {min_budget}', fontsize=fontsize)

    plt.legend()

    plt.savefig(os.path.join(save_folder,f'Quickfilter_{cost_model}.png'), bbox_inches='tight')
    # plt.show()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default='Facebook', help="Name of the dataset to be used (default: 'Facebook')")
    parser.add_argument("--cost_model",type=str,default='degree',help='model of node weights')
    parser.add_argument('--max_budget', type = int ,default=100, help = 'Maximum Budget')
    parser.add_argument('--min_budget', type = int ,default=10, help = 'Minimum Budget')

    parser.add_argument("--delta", type=float, default=0.1, help="Delta")
    parser.add_argument("--eps",type =float,default=1,help="Epsilon")

    args = parser.parse_args()

    
    dataset = args.dataset
    cost_model = args.cost_model
    max_budget = args.max_budget
    min_budget = args.min_budget
    delta = args.delta 
    eps = args.eps

    quickfilter_multi(dataset, cost_model , max_budget, min_budget,delta ,eps,args)

