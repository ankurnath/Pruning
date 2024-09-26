from utils import *
import matplotlib.pyplot as plt
from greedy import gain_adjustment,get_gains,calculate_spread
from helper_functions import *
from knapsack_greedy import knapsack_greedy





def quickfilter_multi(dataset, cost_model , max_budget, min_budget,delta ,eps,num_rr):

    load_graph_file_path=f'../../data/snap_dataset/{dataset}.txt'
    graph = load_graph(load_graph_file_path)
    node_weights = generate_node_weights(graph=graph,cost_model=cost_model)
    # node_weights = np.array(list(node_weights.values()))


    start = time.time()

    gains,node_rr_set,RR = get_gains(graph,num_rr)


    #..............................
    u_taus = {}
    gains_taus ={}
    covered_rr_set_taus = {}
    
    m = int(np.ceil (np.log(max_budget/min_budget)/np.log(1+eps)))
    print ('m =',m)
    curr_obj_taus = defaultdict(int)
    for i in range(m+1):
        tau = (1+eps)**i * min_budget
        u_taus [i] =set([])

        gains_taus[i] = gains.copy()
        covered_rr_set_taus[i] = set ()
        
    for node in graph.nodes():
        for i in range(m+1):
            tau = (1+eps)**i * min_budget
            if gains_taus[i][node]/node_weights[node]>=(delta/tau)*curr_obj_taus[i]:
                curr_obj_taus[i]+=gains_taus[i][node]
                u_taus [i].add(node)
                gain_adjustment(gains=gains_taus[i],node_rr_set=node_rr_set,
                                RR=RR,selected_element=node,covered_rr_set=covered_rr_set_taus[i])
            

    
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
    # gains,_,_ = get_gains(graph,num_rr)
    gains_single = gains.copy()
    curr_obj=0
    pruned_universe_single=[]
    covered_rr_set = set ()
    for node in graph.nodes():

        if gains_single [node]/node_weights[node]>=delta/max_budget*curr_obj:
            curr_obj+=gains_single [node]
            pruned_universe_single.append(node)

            # gains adjustment
            gain_adjustment(gains=gains_single,node_rr_set=node_rr_set,
                            RR=RR,selected_element=node,covered_rr_set=covered_rr_set)
    
            # gain_adjustment(graph,gains,node,uncovered)   
    

    Pg_single = round(len(pruned_universe_single)/graph.number_of_nodes(),4)*100
    print(f'Single budget Size of Pruned universe:{len(pruned_universe_single)}')
    print("Single budget Pruned Universe in percentage:",Pg_single)

    end= time.time()

    timetoprune_single = end - start
    
    
    # budgets = [(1+eps)**i * min_budget for i in range(m+1)] + [max_budget]
    # budgets.sort()

    # if budgets[-1]>max_budget:
    #     budgets.pop()
    # sprint('Budgets',budgets)
    



    df = defaultdict(list)
    step = 20
    budgets = list(range(min_budget,max_budget,step)) +[max_budget]
    sprint(budgets)


    for budget in budgets:

        # print(i)

        start = time.time()

        
        solution_multi_pruned,queries_multi_pruned = knapsack_greedy     (graph=graph,
                                                                          ground_set = pruned_universe_multi, 
                                                                          num_rr=num_rr,
                                                                          budget = budget, 
                                                                          node_weights = node_weights,
                                                                          gains=gains.copy(),
                                                                          node_rr_set=node_rr_set,
                                                                          RR=RR)
        
        # sprint(solution_multi_pruned)
        # raise ValueError('stop')
        # sprint(objective_multi_pruned)
        objective_multi_pruned = calculate_spread(graph=graph,
                                                  solution = solution_multi_pruned)
        
        
        end = time.time()

        time_multi_pruned = end -start

        start = time.time()

        solution_single_pruned,queries_single_pruned =   knapsack_greedy (graph=graph,
                                                                          ground_set = pruned_universe_single, 
                                                                          num_rr=num_rr,
                                                                          budget = budget, 
                                                                          node_weights = node_weights,
                                                                          gains=gains.copy(),
                                                                          node_rr_set=node_rr_set,
                                                                          RR=RR)
        objective_single_pruned = calculate_spread(graph=graph,solution = solution_single_pruned)

       
        
        end = time.time()
        time_single_pruned = end -start

        start = time.time()
        solution_unpruned,queries_unpruned = knapsack_greedy (graph=graph,ground_set =None, 
                                                              num_rr=num_rr,budget = budget, 
                                                              node_weights = node_weights,
                                                              gains=gains.copy(),
                                                              node_rr_set=node_rr_set,
                                                              RR=RR)
        objective_unpruned = calculate_spread(graph=graph,solution=solution_unpruned )

        
        end = time.time()

        time_unpruned = end- start
        sprint(objective_multi_pruned)
        sprint(objective_single_pruned)
        sprint(objective_unpruned)
        
        df['Dataset'].append(dataset)
        df['Budget'].append(budget)
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
        df['Queries (Unpruned)'].append(queries_unpruned)
        df['Queries Multi (pruned)'].append(queries_multi_pruned)
        df['Queries Single (pruned)'].append(queries_single_pruned)

        df['Pruned Ground set Multi(%)'].append(Pg_multi)
        df['Pruned Ground set Single(%)'].append(Pg_single)

        df['Ratio Multi'].append(round(objective_multi_pruned/objective_unpruned,4)*100)
        df['Ratio Single'].append(round(objective_single_pruned/objective_unpruned,4)*100)

        df['Queries Multi(%)'].append(round(queries_multi_pruned/queries_unpruned,4)*100)
        df['Queries Single(%)'].append(round(queries_single_pruned/queries_unpruned,4)*100)

        df['TimeRatio(Multi)'].append(time_multi_pruned/time_unpruned)
        df['TimeRatio(Single)'].append(time_single_pruned/time_unpruned)

        df['TimeToPrune(Multi)'].append(timetoprune_multi)
        df['TimeToPrune(Single)'].append(timetoprune_single)




    df = pd.DataFrame(df)
    print(df)

    save_folder = f'data/{dataset}/knapsack_multi'
    os.makedirs(save_folder,exist_ok=True)
    save_file_path = os.path.join(save_folder,f'Quickfilter_{cost_model}')
    save_to_pickle(df,save_file_path)


    print(df[['Ratio Multi','Ratio Single']])

        
    fontsize = 20
    plt.plot(budgets, df['Ratio Multi'], linestyle='--', marker='o', markersize=20, color='blue', markeredgecolor='black', alpha=0.7, label=f'Multi-Budget {Pg_multi:.2f}%')
    plt.plot(budgets, df['Ratio Single'], linestyle='--', marker='*', markersize=20, color='red', markeredgecolor='black', alpha=0.7, label=f'Single-Budget {Pg_single:.2f}%')
    
    
    plt.xlabel('Budgets', fontsize=fontsize )
    plt.ylabel('Ratios (%)', fontsize=fontsize)
    plt.title(f' Dataset:{args.dataset} Eps:{eps} Delta:{delta} Max Budget:{max_budget} Min Budget: {min_budget}',fontsize=fontsize)
    plt.legend()

    plt.savefig(os.path.join(save_folder,f'Quickfilter_{cost_model}.png'), bbox_inches='tight')
    # plt.show()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default='Facebook', help="Name of the dataset to be used (default: 'Facebook')")
    parser.add_argument("--cost_model",type=str,default='degree',help='model of node weights')
    parser.add_argument('--max_budget', type = int ,default=100, help = 'Maximum Budget')
    parser.add_argument('--min_budget', type = int ,default=10, help = 'Minimum Budget')

    parser.add_argument("--delta", type=float, default=0.5, help="Delta")
    parser.add_argument("--eps",type =float,default=1,help="Epsilon")
    parser.add_argument("--num_rr", type=int, default= 100000  , help="Number of RR sets")

    args = parser.parse_args()

    
    dataset = args.dataset
    cost_model = args.cost_model
    max_budget = args.max_budget
    min_budget = args.min_budget
    delta = args.delta 
    eps = args.eps
    num_rr = args.num_rr

    sprint(dataset)
    sprint(cost_model)
    sprint(max_budget)
    sprint(min_budget)
    sprint(delta)

    quickfilter_multi(dataset=dataset, cost_model=cost_model , 
                      max_budget=max_budget, 
                      min_budget =min_budget,
                      delta= delta ,
                      eps=eps,
                      num_rr=num_rr)

