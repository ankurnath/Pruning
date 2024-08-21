from argparse import ArgumentParser
from utils import *
import pandas as pd
from collections import defaultdict
import numpy as np

from greedy import greedy,gain_adjustment,get_gains
import matplotlib.pyplot as plt



def quickfilter_multi(dataset,budgets,delta,eps):
    load_graph_file_path=f'../../data/test/{dataset}'
    graph=load_from_pickle(load_graph_file_path)

    # load_graph_file_path=f'../../data/snap_dataset/{dataset}.txt'
    # graph=nx.read_edgelist(f'../../data/snap_dataset/{dataset}.txt', create_using=nx.Graph(), nodetype=int)

    # pruning stage

    # gains={node:graph.degree(node) for node in graph.nodes()}
    df=defaultdict(list)

    
    for budget in budgets:

        u_taus = {}
        gains_taus ={}
        uncovered_taus = {}
        

        m = int(np.floor (np.log(budget)/np.log(1+eps)))
        print ('m =',m)
        curr_obj_taus = defaultdict(int)
        for i in range(m+1):
            tau = (1+eps)**i 
            # print(tau)
            u_taus [i] =set([])
            gains_taus [i] = get_gains(graph,ground_set=None)
            uncovered_taus[i]=defaultdict(lambda: True)
            
        for node in graph.nodes():

            for i in range(m+1):
                tau = (1+eps)**i 

                if gains_taus[i][node]>=(delta/tau)*curr_obj_taus[i]:
                    # print(gains_taus[i][node])
                    curr_obj_taus[i]+=gains_taus[i][node]
                    u_taus [i].add(node)
                    # pruned_universe.append(node)

                    # gains adjustment
                    gain_adjustment(graph,gains_taus[i],node,uncovered_taus[i])
                

        
        for key in u_taus:
            print(f'key:{key} tau:{(1+eps)**key} size:{len(u_taus[key])}')


        u = u_taus [0]

        for i in range(1,m+1):
            u = u.union(u_taus[i])

        pruned_universe = list(u)
        
        # Pg=1-len(pruned_universe)/graph.number_of_nodes()
        Pg=len(pruned_universe)/graph.number_of_nodes()
        print("Pg:",round(Pg,4)*100)

        print('Pruned Universe:',len(pruned_universe))
        
        
        # # Subgraph 
        # subgraph =make_subgraph(graph,pruned_universe)


        ratios = []

        x = [int((1+eps)**i)  for i in range(m+1)] + [budget]
        print(x)

        for i in x:
            solution_subgraph,_ = greedy(graph,i,pruned_universe)
            greedy_solution,_ = greedy(graph,i)
            coverage= calculate_cover(graph,solution_subgraph)
            ratios.append(coverage/calculate_cover(graph,greedy_solution))

            
        fontsize = 20
        # plt.plot(x,ratios)
        # plt.scatter(x, ratios, color='blue', marker='o', s=100, edgecolor='black', alpha=0.7)
        plt.plot(x, ratios, linestyle='--', marker='o', markersize=20, color='blue', markeredgecolor='black', alpha=0.7, label='Data Points')
        plt.xlabel('Budgets', fontsize=fontsize )
        plt.ylabel('Ratios', fontsize=fontsize)
        plt.title(f'Dataset:{dataset}(Test), Eps:{eps} Delta:{delta} Size Constraint:{budget}',fontsize=fontsize)
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
    parser.add_argument("--budgets", nargs='+', type=int, help="Budgets")
    parser.add_argument("--delta", type=float, default=0.1, help="Delta")
    parser.add_argument("--eps",type=float,default=0.1,help="Epsilon")

    args = parser.parse_args()


    

    quickfilter_multi(dataset=args.dataset,budgets=args.budgets,delta=args.delta,eps=args.eps)
