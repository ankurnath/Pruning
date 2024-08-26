from utils import *
from greedy import greedy,gain_adjustment,get_gains

from knapsack_numba_greedy import knapsack_numba_greedy

# from budgeted_greedy import modified_greedy
# import matplotlib.pyplot as plt
from IP_solver import gurobi_solver



def quickfilter_random(dataset, cost_model , max_budget, min_budget,delta ,eps,args):

    load_graph_file_path=f'../../data/snap_dataset/{dataset}.txt'
    graph = load_graph(load_graph_file_path)
    node_weights = generate_node_weights(graph=graph,cost_model=cost_model)
   
    start = time.time()

    m = int(np.floor (np.log(max_budget/min_budget)/np.log(1+eps)))
    print ('m =',m)
    curr_obj_taus = defaultdict(int)
    gains = get_gains(graph=graph,ground_set=None)

    uncovered = defaultdict(lambda: True)

    taus = [ (1+eps)**i * min_budget for i in range(m+1)]

    curr_obj = 0
    pruned_universe_multi = []    
    for node in graph.nodes():
        
        tau = random.choice(taus)

        if gains[node]/node_weights[node]>=(delta/tau)*curr_obj:

            curr_obj += gains[node]
            pruned_universe_multi.append(node)

            gain_adjustment(graph,gains,node,uncovered)


    Pg_multi=len(pruned_universe_multi)/graph.number_of_nodes()
    Pg_multi = round(Pg_multi,4)*100
    print("Pg(%):",Pg_multi)
    print('Multi budget Pruned Universe:',len(pruned_universe_multi))
    print("Multi budget Pruned Universe in percentage:",Pg_multi)

    end = time.time()

    timetoprune_multi = end-start



    start = time.time()
    gains=get_gains(graph,ground_set=None)
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
    print("Single budget Pruned Universe in percentage:",round(len(pruned_universe_single)/graph.number_of_nodes(),4)*100)
    
    end= time.time()

    timetoprune_single = end - start


    

    # x = [int((1+eps)**i * min_budget)  for i in range(m+1)] + [max_budget]

    x = [(1+eps)**i * min_budget for i in range(m+1)] + [max_budget]
    x.sort()

    if x[-1]>max_budget:
        x.pop()
    print('Budgets',x)
    subgraph_multi = make_subgraph(graph,pruned_universe_multi)
    subgraph_single = make_subgraph(graph,pruned_universe_single)

    # print(subgraph_multi.number_of_nodes()/graph.number_of_nodes())

    df = defaultdict(list)

    for i in x[::-1]:
    # for i in range(10,110,10):
        sprint(i)
        start = time.time()
        objective_multi_pruned,queries_multi_pruned,solution_multi_pruned= gurobi_solver(        graph= subgraph_multi, 
                                                                                                 budget=i,
                                                                                                 node_weights=node_weights)
        end = time.time()

        time_multi_pruned = end -start

        start = time.time()
        objective_single_pruned,queries_single_pruned,solution_single_pruned= gurobi_solver(graph = subgraph_single, 
                                                                                                 budget=i,
                                                                                                 node_weights=node_weights
                                                                                                 )
        
        end = time.time()
        time_single_pruned = end -start

        start = time.time()
        objective_unpruned,queries_unpruned,solution_unpruned= gurobi_solver(graph=graph,budget=i,
                                                                                 node_weights=node_weights)
        
        
        end = time.time()
        
        sprint(objective_single_pruned)
        sprint(objective_unpruned)

        time_unpruned = end- start
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

    save_folder = f'data/{dataset}/knapsack_multi_random'
    os.makedirs(save_folder,exist_ok=True)
    save_file_path = os.path.join(save_folder,f'Quickfilter_{cost_model}')
    save_to_pickle(df,save_file_path)


    print(df[['Budget','Ratio Multi','Queries Multi (pruned)','Queries Multi(%)','Ratio Single','Queries Single (pruned)','Queries Single(%)']])

        
    fontsize = 20
    plt.plot(x, df['Ratio Multi'], linestyle='--', marker='o', markersize=20, color='blue', markeredgecolor='black', alpha=0.7, label=f'Multi-Budget {Pg_multi:.2f}%')
    plt.plot(x, df['Ratio Single'], linestyle='--', marker='*', markersize=20, color='red', markeredgecolor='black', alpha=0.7, label=f'Single-Budget {Pg_single:.2f}%')
    
    
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

    parser.add_argument("--delta", type=float, default=0.1, help="Delta")
    parser.add_argument("--eps",type =float,default=1,help="Epsilon")

    args = parser.parse_args()

    
    dataset = args.dataset
    cost_model = args.cost_model
    max_budget = args.max_budget
    min_budget = args.min_budget
    delta = args.delta 
    eps = args.eps

    quickfilter_random(dataset, cost_model , max_budget, min_budget,delta ,eps,args)

        



        
    
