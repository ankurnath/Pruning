from utils import *

from greedy import greedy,gain_adjustment,get_gains

from knapsack_numba_greedy import knapsack_numba_greedy
import matplotlib.pyplot as plt
from helper_functions import calculate_obj
# from IP_solver import gurobi_solver

def qs(graph,node_weights,budget,delta,eps):
    start = time.time()
    gains = get_gains(graph,ground_set=None)
    curr_obj = 0
    queries_to_prune = 0
    # pruned_universe=[] 
    a = set()
    # a_start = set() 
    a_start = max(gains, key=gains.get)
    # sprint(a_start)
    a_s = set()
    

    obj_a_s = 0
    uncovered=defaultdict(lambda: True)

    N = graph.number_of_nodes()
    for node in tqdm(graph.nodes()):
        if node_weights[node] >= max_budget:
            continue
        queries_to_prune += 1
        if gains[node]/node_weights[node] >= delta/budget*curr_obj:
            curr_obj+=gains[node]
            # pruned_universe.append(node)
            a.add(node)
            gain_adjustment(graph,gains,node,uncovered)


        ### New addition
        if curr_obj > N/eps*obj_a_s:
            # print('This happened')
            
            # a = a.difference(a_s)
            a.difference_update(a_s)
            a_s = a.copy()

            obj_a_s = calculate_obj(graph=graph,solution=a_s)
            curr_obj = obj_a_s
            queries_to_prune +=1
            
    end= time.time()

    time_to_prune = end-start

    # print('time elapsed to pruned',time_to_prune)
    a.add(a_start)
    pruned_universe = list(a)
    return pruned_universe,queries_to_prune,time_to_prune



def quickfilter_multi(dataset, cost_model , max_budget, min_budget,delta ,eta,eps,args):

    load_graph_file_path=f'../../data/snap_dataset/{dataset}.txt'
    graph = load_graph(load_graph_file_path)
    N= graph.number_of_nodes()
    node_weights = generate_node_weights(graph=graph,cost_model=cost_model)


    start = time.time()

    # m = int(np.floor (np.log(max_budget/min_budget)/np.log(1+eta)+1))

    pruned_universe_multi =[]

    high = int(np.log(min_budget/max_budget)/np.log(1-eta) +1 )
    low = int(np.log(max_budget/max_budget)/np.log(1-eta))
    # for i in range(m+1):
    # i=0
    # while True:
    for i in range(low,high+1):
        # tau = (1+eta)**i * min_budget
        
        tau = max_budget*(1-eta)**i
        sprint(tau)
        pruned_universe,queries_to_prune,time_to_prune = qs(graph=graph,budget=tau,node_weights=node_weights,delta=delta,eps=eps)
        pruned_universe_multi +=pruned_universe
    # u_taus = {}

    # u_taus_prime = {}
    
    # gains_taus ={}
    # uncovered_taus = {}
    
    # m = int(np.floor (np.log(max_budget/min_budget)/np.log(1+eta)+1))
    # print ('m =',m)
    # curr_obj_taus = defaultdict(int)
    # curr_obj_taus_prime = defaultdict(int)
    # for i in range(m+1):
    #     tau = (1+eta)**i * min_budget
    #     u_taus [i] =set([])

    #     ###
    #     u_taus_prime [i] =set([])

    #     gains_taus [i] = get_gains(graph,ground_set=None)
    #     uncovered_taus[i] = defaultdict(lambda: True)
        
    # for node in graph.nodes():
    #     for i in range(m+1):
    #         tau = (1+eta)**i * min_budget
    #         if node_weights[node] >= tau:
    #             continue
    #         if gains_taus[i][node]/node_weights[node]>=(delta/tau)*curr_obj_taus[i]:
    #             curr_obj_taus[i]+=gains_taus[i][node]
    #             u_taus [i].add(node)
    #             # gains adjustment
    #             gain_adjustment(graph,gains_taus[i],node,uncovered_taus[i])

    #         if curr_obj_taus[i]> N/eps*curr_obj_taus_prime[i]:
    #             u_taus[i] =u_taus[i].difference(u_taus_prime [i])
    #             u_taus_prime [i] =  u_taus[i]
    #             curr_obj_taus[i] = calculate_obj(graph=graph,solution=u_taus[i])
    #             curr_obj_taus_prime[i] = curr_obj_taus[i]
    
    # # for key in u_taus:
    # #     print(f'key:{key} tau:{int((1+eta)**key * min_budget)} size:{len(u_taus[key])}')

    # for key in u_taus:
    #     print(f'key:{key} tau:{(1+eta)**key * min_budget} size:{len(u_taus[key])}')


    # u = u_taus [0]

    # for i in range(1,m+1):
    #     u = u.union(u_taus[i])

    # pruned_universe_multi = list(u)

    pruned_universe_multi = set(pruned_universe_multi)
    end = time.time()

    timetoprune_multi = end-start

    Pg_multi=len(pruned_universe_multi)/graph.number_of_nodes()
    Pg_multi = round(Pg_multi,4)*100
    print("Pg(%):",Pg_multi)
    print('Multi budget Pruned Universe:',len(pruned_universe_multi))
    print("Multi budget Pruned Universe in percentage:",Pg_multi)



    start = time.time()
    # gains = get_gains(graph,ground_set=None)
    # curr_obj=0
    # pruned_universe_single=[]
    # uncovered=defaultdict(lambda: True)
    # for node in graph.nodes():
    #     if node_weights[node] >= max_budget:
    #         continue

    #     if gains[node]/node_weights[node]>=delta/max_budget*curr_obj:
    #         curr_obj+=gains[node]
    #         pruned_universe_single.append(node)

    #         # gains adjustment
    #         gain_adjustment(graph,gains,node,uncovered)   
    
    pruned_universe_single,_,_ = qs(graph=graph,node_weights=node_weights,budget=max_budget,delta=delta,eps=eps)
    Pg_single = round(len(pruned_universe_single)/graph.number_of_nodes(),4)*100
    print(f'Single budget Size of Pruned universe:{len(pruned_universe_single)}')
    print("Single budget Pruned Universe in percentage:",Pg_single)

    end= time.time()

    timetoprune_single = end - start
    
    


    df = defaultdict(list)
    step = 20
    budgets = list(range(min_budget,max_budget,step)) +[max_budget]
    sprint(budgets)

    save_folder = f'data/{dataset}/knapsack_multi'
    os.makedirs(save_folder,exist_ok=True)
    save_file_path = os.path.join(save_folder,f'Quickfilter_{cost_model}')
    
    for i in budgets:

        start = time.time()
        objective_multi_pruned,queries_multi_pruned,solution_multi_pruned= knapsack_numba_greedy(graph=graph, 
                                                                                                 budget=i,node_weights=node_weights,
                                                                                                 ground_set=pruned_universe_multi)
        end = time.time()

        time_multi_pruned = end -start

        start = time.time()
        objective_single_pruned,queries_single_pruned,solution_single_pruned= knapsack_numba_greedy(graph=graph, 
                                                                                                 budget=i,node_weights=node_weights,
                                                                                                 ground_set=pruned_universe_single)
        
        end = time.time()
        time_single_pruned = end -start

        start = time.time()

        try:
            previous_df = load_from_pickle(save_file_path)

            objective_unpruned = previous_df[previous_df['Budget']==i]['Objective Value(Unpruned)'].iloc[0]
            queries_unpruned = previous_df[previous_df['Budget']==i]['Queries (Unpruned)'].iloc[0]
            print('Loaded all data from previous run')
        except:

            objective_unpruned,queries_unpruned,solution_unpruned= knapsack_numba_greedy(graph=graph,budget=i,
                                                                                 node_weights=node_weights)
        
        
        end = time.time()

        time_unpruned = end- start
        ### TOP-K
        gains = get_gains(graph,ground_set=None)
        density_gain = {node: gains[node]/node_weights[node] for node in gains}
        pruned_universe_multi_top_k = [key for key, _ in sorted(density_gain.items(), 
                                key=lambda item: item[1], reverse=True) if node_weights[key] <= i] [:len(pruned_universe_multi)]
        
        objective_multi_top_k,_,_ = knapsack_numba_greedy(graph=graph,budget=i,
                                                        node_weights=node_weights,
                                                        ground_set=pruned_universe_multi_top_k)
        
        pruned_universe_single_top_k = [key for key, _ in sorted(density_gain.items(), 
                                key=lambda item: item[1], reverse=True) if node_weights[key] <= i ] [:len(pruned_universe_single)]
        
        objective_single_top_k,_,_ = knapsack_numba_greedy(graph=graph,budget=i,
                                                        node_weights=node_weights,
                                                        ground_set=pruned_universe_single_top_k )
        







        df['Dataset'].append(dataset)
        df['Budget'].append(i)
        df['Delta'].append(delta)
        df['Eps'].append(eps)
        df['Eta'].append(eta)
        df['Objective Value(Unpruned)'].append(objective_unpruned)
        df['Objective Value Multi(Pruned)'].append(objective_multi_pruned)
        df['Objective Value Single(Pruned)'].append(objective_single_pruned)
        df['Ground Set'].append(graph.number_of_nodes())
        df['Ground set Multi (Pruned)'].append(len(pruned_universe_multi))
        df['Ground set Single (Pruned)'].append(len(pruned_universe_single))
        
        df['Time(Unpruned)'].append(time_unpruned)
        df['Time Multi(Pruned)'].append(time_multi_pruned)
        df['Time Single(Pruned)'].append(time_single_pruned)
        df['Queries (Unpruned)'].append(queries_unpruned)
        df['Queries Multi (pruned)'].append(queries_multi_pruned)
        df['Queries Single (pruned)'].append(queries_single_pruned)

        df['Pruned Ground set Multi(%)'].append(Pg_multi)
        df['Pruned Ground set Single(%)'].append(Pg_single)

        df['Ratio Multi'].append(round(objective_multi_pruned/objective_unpruned,4)*100)
        df['Ratio Single'].append(round(objective_single_pruned/objective_unpruned,4)*100)
        ### TOP-K
        df['Ratio Multi(TOP-K)'].append(round(objective_multi_top_k/objective_unpruned,4)*100)
        df['Ratio Single(TOP-K)'].append(round(objective_single_top_k/objective_unpruned,4)*100)


        df['Queries Multi(%)'].append(round(queries_multi_pruned/queries_unpruned))
        df['Queries Single(%)'].append(round(queries_single_pruned/queries_unpruned))

        df['TimeRatio(Multi)'].append(time_multi_pruned/time_unpruned)
        df['TimeRatio(Single)'].append(time_single_pruned/time_unpruned)

        df['TimeToPrune(Multi)'].append(timetoprune_multi)
        df['TimeToPrune(Single)'].append(timetoprune_single)




    df = pd.DataFrame(df)

    
    save_to_pickle(df,save_file_path)


    print(df[['Ratio Multi','Ratio Multi(TOP-K)','Ratio Single','Ratio Single(TOP-K)']])

        
    fontsize = 20
    plt.plot(budgets, df['Ratio Multi'], linestyle='--', marker='o', markersize=20, color='blue', markeredgecolor='black', alpha=0.7, label=f'Multi-Budget {Pg_multi:.2f}%')
    plt.plot(budgets, df['Ratio Single'], linestyle='--', marker='*', markersize=20, color='red', markeredgecolor='black', alpha=0.7, label=f'Single-Budget {Pg_single:.2f}%')
    plt.plot(budgets, df['Ratio Multi(TOP-K)'], linestyle='--', marker='^', markersize=20, color='blue', markeredgecolor='black', alpha=0.7, label=f'Multi-Budget (TOP-K) {Pg_multi:.2f}%')
    plt.plot(budgets, df['Ratio Single(TOP-K)'], linestyle='--', marker='s', markersize=20, color='red', markeredgecolor='black', alpha=0.7, label=f'Single-Budget (TOP-K) {Pg_single:.2f}%')
    
    plt.xlabel('Budgets', fontsize=fontsize )
    plt.ylabel('Ratios (%)', fontsize=fontsize)
    plt.title(f' Dataset:{args.dataset} Cost model:{args.cost_model} Delta:{delta} eps:{eps} eta:{eta}\n Max Budget:{max_budget} Min Budget: {min_budget}',fontsize=10)
    plt.legend()

    plt.savefig(os.path.join(save_folder,f'Quickfilter_{cost_model}.png'), bbox_inches='tight')
    # plt.show()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default='Facebook', help="Name of the dataset to be used (default: 'Facebook')")
    parser.add_argument("--cost_model",type=str,default='aistats',help='model of node weights')
    parser.add_argument('--max_budget', type = int ,default=100, help = 'Maximum Budget')
    parser.add_argument('--min_budget', type = int ,default=10, help = 'Minimum Budget')

    parser.add_argument("--delta", type=float, default=0.5, help="Delta")
    parser.add_argument("--eps", type=float, default=0.1, help="Eps")
    parser.add_argument("--eta",type =float,default=0.5,help="Eta")

    args = parser.parse_args()

    
    dataset = args.dataset
    cost_model = args.cost_model
    max_budget = args.max_budget
    min_budget = args.min_budget
    delta = args.delta 
    eps = args.eps
    eta = args.eta

    quickfilter_multi(dataset, cost_model , max_budget, min_budget,delta=delta ,eps=eps,eta=eta,args=args)

