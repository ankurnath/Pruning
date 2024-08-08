from argparse import ArgumentParser
from utils import *
import pandas as pd
from collections import defaultdict
import numpy as np

from greedy import greedy,gain_adjustment,get_gains

from budgeted_greedy import modified_greedy
import matplotlib.pyplot as plt



def quickfilter_multi(graph, node_weights , max_budget, min_budget,delta ,eps,args):
   
    df=defaultdict(list)
 
    # for budget in budgets:

    u_taus = {}
    gains_taus ={}
    uncovered_taus = {}
    

    m = int(np.floor (np.log(max_budget/min_budget)/np.log(1+eps)))
    print ('m =',m)
    curr_obj_taus = defaultdict(int)
    for i in range(m+1):
        tau = (1+eps)**i * min_budget
        u_taus [i] =set([])
        gains_taus [i] = get_gains(graph,ground_set=None)
        uncovered_taus[i] = defaultdict(lambda: True)
        
    for node in graph.nodes():

        for i in range(m+1):
            # print('Do we cast tau to integer ?')
            tau = int((1+eps)**i * min_budget)

            if gains_taus[i][node]/node_weights[node]>=(delta/tau)*curr_obj_taus[i]:
                # print(gains_taus[i][node])
                curr_obj_taus[i]+=gains_taus[i][node]
                u_taus [i].add(node)
                # pruned_universe.append(node)

                # gains adjustment
                gain_adjustment(graph,gains_taus[i],node,uncovered_taus[i])
            

    
    for key in u_taus:
        print(f'key:{key} tau:{int((1+eps)**key * min_budget)} size:{len(u_taus[key])}')


    u = u_taus [0]

    for i in range(1,m+1):
        u = u.union(u_taus[i])

    pruned_universe = list(u)
    
    # Pg=1-len(pruned_universe)/graph.number_of_nodes()
    Pg=len(pruned_universe)/graph.number_of_nodes()
    # print("Pg:",round(Pg,4)*100)

    print('Multi budget Pruned Universe:',len(pruned_universe))
    print("Multi budget Pruned Universe in percentage:",round(Pg,4)*100)

    df['Pruned Ground (Multi)%'].append(round(Pg,4)*100)
    
    
    # # Subgraph 
    # subgraph =make_subgraph(graph,pruned_universe)


    multi_ratios = []

    multi_ratios_covers= []

    x = [int((1+eps)**i * min_budget)  for i in range(m+1)] + [max_budget]
    x.sort()

    if x[-1]>max_budget:
        x.pop()
    print(x)

    greedy_solutions = []
    greedy_covers = []
    for i in x:
        solution_subgraph,_ = modified_greedy(graph=graph, budget=i,node_weights=node_weights,ground_set=pruned_universe) 
        greedy_solution,_ = modified_greedy(graph=graph, budget=i, node_weights=node_weights)
        greedy_solutions.append(greedy_solution) 
        coverage = calculate_cover(graph,solution_subgraph)
        multi_ratios_covers .append(coverage)
        greedy_cover = calculate_cover(graph,greedy_solution)
        greedy_covers.append(greedy_cover)
        multi_ratios.append(coverage/greedy_cover )
        

    df['Budgets'] = [x ]
    df['Cover (Multi)'] = [multi_ratios_covers]
    df['Ratio (Multi)'] = [multi_ratios]
    df['Cover (greedy)'] = [greedy_covers]
    df['Solutions (greedy)'] = [greedy_solutions]

    #################################################
    gains=get_gains(graph,ground_set=None)
    curr_obj=0
    pruned_universe=[]
    uncovered=defaultdict(lambda: True)
    for node in graph.nodes():

        if gains[node]/node_weights[node]>=delta/max_budget*curr_obj:
            curr_obj+=gains[node]
            pruned_universe.append(node)

            # gains adjustment
            gain_adjustment(graph,gains,node,uncovered)    


    print(f'Single budget Size of Pruned universe:{len(pruned_universe)}')
    print("Single budget Pruned Universe in percentage:",round(len(pruned_universe)/graph.number_of_nodes(),4)*100)
    
    df['Pruned Ground (Single)%'].append(round(Pg,4)*100)
    
    single_ratios = []
    single_ratios_covers= []
    for idx,i in enumerate(x):
        solution_subgraph,_ = modified_greedy(graph=graph, budget=i,node_weights=node_weights,ground_set = pruned_universe) 
        # greedy_solution,_ = modified_greedy(graph=graph, budget=i, node_weights=node_weights) 
        coverage= calculate_cover(graph,solution_subgraph)
        single_ratios_covers.append(coverage)
        single_ratios.append(coverage/calculate_cover(graph,greedy_solutions[idx]))

    df['Cover (Single)'] = [single_ratios_covers]
    df['Ratio (Single)'] = [single_ratios]
    #################################################
    # print(df)
    df=pd.DataFrame(df)
    print(df)

        
    fontsize = 20
    # plt.plot(x,ratios)
    # plt.scatter(x, ratios, color='blue', marker='o', s=100, edgecolor='black', alpha=0.7)
    # plt.plot(x,np.array(multi_ratios_covers)/np.array(single_ratios_covers),linestyle='--', marker='o', markersize=20, color='blue', markeredgecolor='black', alpha=0.7, label='Multi-Budget vs Single-Budget Ratios')
    # plt.plot(x, multi_ratios_covers, linestyle='--', marker='o', markersize=20, color='blue', markeredgecolor='black', alpha=0.7, label='Multi-Budget')
    # plt.plot(x, single_ratios_covers, linestyle='--', marker='*', markersize=20, color='red', markeredgecolor='black', alpha=0.7, label='Single-Budget')
    plt.plot(x, multi_ratios, linestyle='--', marker='o', markersize=20, color='blue', markeredgecolor='black', alpha=0.7, label='Multi-Budget')
    plt.plot(x, single_ratios, linestyle='--', marker='*', markersize=20, color='red', markeredgecolor='black', alpha=0.7, label='Single-Budget')
    
    # plt.plot(x, ratios, linestyle='--', marker='o', markersize=20, color='blue', markeredgecolor='black', alpha=0.7, label='Multi-Budget')
    plt.xlabel('Budgets', fontsize=fontsize )
    plt.ylabel('Ratios', fontsize=fontsize)
    plt.title(f' Dataset:{args.dataset} (Test) Eps:{eps} Delta:{delta} Max Budget:{max_budget} Min Budget: {min_budget}',fontsize=fontsize)
    plt.legend()
    plt.show()

    print('Multi-Budget ratios',multi_ratios)
    print('Single Budget ratios',single_ratios)
    df['dataset'] = args.dataset
    df['Eps'] = eps
    df['Delta'] = delta
    df['Max Budget'] = max_budget
    df['Min Budget'] = min_budget


        
    #     Pv=1-subgraph.number_of_nodes()/graph.number_of_nodes()
    #     Pe=1-subgraph.number_of_edges()/graph.number_of_edges()
        
    #     solution_subgraph,_ = greedy(graph,budget,pruned_universe)

    #     greedy_solution,_ = greedy(graph,budget)
    #     # print()

    #     coverage= calculate_cover(graph,solution_subgraph)

    #     df['Budget'].append(budget)
    #     df['Pv'].append(Pv)
    #     df['Pe'].append(Pe)
    #     df['Pg'].append(Pg)
    #     df['Objective Value'].append(coverage)
    #     df['Ratio'].append(coverage/calculate_cover(graph,greedy_solution))
    #     df['Solution'].append(solution_subgraph)
    #     df['Objective Value (Ratio)'].append(coverage/graph.number_of_nodes())
        

    # df['delta']=[delta]*len(df['Budget'])

    # df=pd.DataFrame(df)

    
    # try:
    #     df['Ratio']=df['Objective Value']/load_from_pickle(f'data/{args.dataset}/Greedy')['Objective Value']
    # except:
    #     raise ValueError('Greedy value is not found.')
    # print(df)


    save_folder=f'data/knapsack'
    file_path=os.path.join(save_folder,args.dataset)
    os.makedirs(save_folder,exist_ok=True)
    save_to_pickle(df,file_path)
    load_from_pickle(file_path=file_path)
    print(df.columns)
    # print(df['Cover (greedy)'])
    # print(df['Solutions (greedy)'][0])
    
    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default='Facebook', help="Name of the dataset to be used (default: 'Facebook')")
    parser.add_argument("--cost_model",type=str,default='random',help='model of node weights')
    parser.add_argument('--max_budget', type = int ,default=100, help = 'Maximum Budget')
    parser.add_argument('--min_budget', type = int ,default=10, help = 'Minimum Budget')

    parser.add_argument("--delta", type=float, default=0.1, help="Delta")
    parser.add_argument("--eps",type =float,default=2,help="Epsilon")

    args = parser.parse_args()

    file_path=f'../../data/test/{args.dataset}'
    graph = load_from_pickle(file_path)

    node_weights = load_from_pickle(f'../../data/test/{args.dataset}_weights_{args.cost_model}')

    max_budget = args.max_budget
    min_budget = args.min_budget
    delta = args.delta 
    eps = args.eps



    quickfilter_multi(graph = graph, node_weights = node_weights, 
                      max_budget = max_budget, min_budget = min_budget, 
                      delta = delta, eps = eps ,args = args)
