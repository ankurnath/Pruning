from argparse import ArgumentParser
from utils import *
import pandas as pd
from collections import defaultdict
import numpy as np

from greedy import greedy,gain_adjustment,get_gains

from numba_greedy import numba_greedy

# from budgeted_greedy import modified_greedy
import matplotlib.pyplot as plt



def quickfilter_multi(graph, node_weights , max_budget, min_budget,delta ,eps,args):
   
    df=defaultdict(list)
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
            # tau = int((1+eps)**i * min_budget)
            tau = (1+eps)**i * min_budget

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

    pruned_universe_multi = list(u)

    Pg=len(pruned_universe_multi)/graph.number_of_nodes()
    print("Pg(%):",round(Pg,4)*100)
    print('Multi budget Pruned Universe:',len(pruned_universe_multi))
    print("Multi budget Pruned Universe in percentage:",round(Pg,4)*100)

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
    
    print(f'Single budget Size of Pruned universe:{len(pruned_universe_single)}')
    print("Single budget Pruned Universe in percentage:",round(len(pruned_universe_single)/graph.number_of_nodes(),4)*100)
    
    multi_ratios = []
    single_ratios = []
    greedy_coverages = []

    x = [int((1+eps)**i * min_budget)  for i in range(m+1)] + [max_budget]
    x.sort()

    if x[-1]>max_budget:
        x.pop()
    print('Budgets',x)

    for i in x:
        # solution_subgraph,_ = modified_greedy(graph=graph, budget=i,node_weights=node_weights,ground_set=pruned_universe) 
        # greedy_solution,_ = modified_greedy(graph=graph, budget=i, node_weights=node_weights) 
        greedy_solution,_ = numba_greedy(graph=graph, budget=i, node_weights=node_weights) 
        solution_subgraph_multi,_ = numba_greedy(graph=graph, budget=i,node_weights=node_weights,ground_set=pruned_universe_multi)
        solution_subgraph_single,_ = numba_greedy(graph=graph, budget=i,node_weights=node_weights,ground_set=pruned_universe_single) 
        greedy_coverage = calculate_cover(graph,greedy_solution)
        greedy_coverages.append(greedy_coverage)
        multi_coverage= calculate_cover(graph,solution_subgraph_multi)
        single_coverage = calculate_cover(graph,solution_subgraph_single)
        multi_ratios.append(multi_coverage/greedy_coverage)
        single_ratios.append(single_coverage /greedy_coverage)

        print('greedy')
        print('Multi-ratio',multi_ratios[-1])
        print('Single-ratio',single_ratios[-1])

    print(greedy_coverages)
    # print('Degree',sorted([graph.degree(node) for node in greedy_solution]))
    
    
    # print(solution_subgraph)
    # print('Degree',[graph.degree(node) for node in solution_subgraph])

    #################################################
     


    
    
    # for i in x:
    #     # solution_subgraph,_ = modified_greedy(graph=graph, budget=i,node_weights=node_weights,ground_set = pruned_universe) 
    #     # greedy_solution,_ = modified_greedy(graph=graph, budget=i, node_weights=node_weights) 
    #     solution_subgraph,_ = numba_greedy(graph=graph, budget=i,node_weights=node_weights,ground_set = pruned_universe) 
    #     greedy_solution,_ = numba_greedy(graph=graph, budget=i, node_weights=node_weights) 
    #     coverage= calculate_cover(graph,solution_subgraph)
    #     single_ratios.append(coverage/calculate_cover(graph,greedy_solution))

    
    #################################################

        
    fontsize = 20
    # plt.plot(x,ratios)
    # plt.scatter(x, ratios, color='blue', marker='o', s=100, edgecolor='black', alpha=0.7)
    plt.plot(x, multi_ratios, linestyle='--', marker='o', markersize=20, color='blue', markeredgecolor='black', alpha=0.7, label='Multi-Budget')
    plt.plot(x, single_ratios, linestyle='--', marker='*', markersize=20, color='red', markeredgecolor='black', alpha=0.7, label='Single-Budget')
    
    # plt.plot(x, ratios, linestyle='--', marker='o', markersize=20, color='blue', markeredgecolor='black', alpha=0.7, label='Multi-Budget')
    plt.xlabel('Budgets', fontsize=fontsize )
    plt.ylabel('Ratios', fontsize=fontsize)
    plt.title(f' Dataset:{args.dataset} (Test) Eps:{eps} Delta:{delta} Max Budget:{max_budget} Min Budget: {min_budget}',fontsize=fontsize)
    plt.legend()
    plt.show()



        
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


    # save_folder=f'data/{args.dataset}'
    # file_path=os.path.join(save_folder,'QuickFilter')
    # os.makedirs(save_folder,exist_ok=True)
    # save_to_pickle(df,file_path) 

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default='Facebook', help="Name of the dataset to be used (default: 'Facebook')")
    parser.add_argument("--cost_model",type=str,default='degree',help='model of node weights')
    parser.add_argument('--max_budget', type = int ,default=100, help = 'Maximum Budget')
    parser.add_argument('--min_budget', type = int ,default=10, help = 'Minimum Budget')

    parser.add_argument("--delta", type=float, default=0.1, help="Delta")
    parser.add_argument("--eps",type =float,default=2,help="Epsilon")

    args = parser.parse_args()

    # file_path=f'../../data/test/{args.dataset}'
    # graph = load_from_pickle(file_path)

    # node_weights = load_from_pickle(f'../../data/test/{args.dataset}_weights_{args.cost_model}')
    
    graph = nx.read_edgelist(f'../../data/snap_dataset/{args.dataset}.txt', create_using=nx.Graph(), nodetype=int)
    
    if args.cost_model == 'uniform':
        node_weights = {node:1 for node in graph.nodes()}

    elif args.cost_model == 'degree':
        # alpha = 1/20
        alpha = 1/20
        out_degrees = {node: graph.degree(node) for node in graph.nodes()}
        out_degree_max = np.max(list(out_degrees.values()))
        out_degree_min = np.min(list(out_degrees.values()))
        node_weights = {node: (out_degrees[node] - out_degree_min + alpha) / (out_degree_max - out_degree_min) for node in graph.nodes()}

    else:
        raise NotImplementedError('Unknown model')
    
    max_budget = args.max_budget
    min_budget = args.min_budget
    delta = args.delta 
    eps = args.eps



    quickfilter_multi(graph = graph, node_weights = node_weights, 
                      max_budget = max_budget, min_budget = min_budget, 
                      delta = delta, eps = eps ,args = args)
