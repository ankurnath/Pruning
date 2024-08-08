from argparse import ArgumentParser
from utils import *
# import pandas as pd
from collections import defaultdict
import numpy as np
import os

from greedy import get_gains, gain_adjustment
# from tqdm import tqdm
from numba import njit

def from_networkx(graph:nx.Graph):


    adj_list = []

    n = graph.number_of_nodes()
    start = [0 for _ in range(n)]
    end = [0 for _ in range(n)]
    
    # idx = 0
    for node in graph.nodes():
        start[node] = len(adj_list)
        neighbors = list(graph.neighbors(node))
        end[node] = start[node] + len(neighbors)
        adj_list += neighbors
        # idx+=1
        
        
    return np.array(adj_list), np.array(start), np.array(end)


@njit 
def get_gains(adj_list,start,end):

    N = start.shape[0]
    gains = np.zeros(N)

    for i in range(N):
        gains[i] = end[i]-start[i]+1

    return gains

@njit
def gain_adjustment(adj_list,start,end,gains,selected_element,uncovered):

    # print('Gain:',gains[selected_element])
    if uncovered[selected_element] ==1 :
        gains[selected_element]-=1
        uncovered[selected_element] = 0
        for neighbor in adj_list[start[selected_element]:end[selected_element]]:
            if gains[neighbor]>0:
                gains[neighbor]-=1

    for neighbor in adj_list[start[selected_element]:end[selected_element]]:
        if uncovered[neighbor] == 1:
            uncovered[neighbor]= 0            
            gains[neighbor]-=1
            for neighbor_of_neighbor in adj_list[start[neighbor]:end[neighbor]]:
                gains[neighbor_of_neighbor]-=1

    return gains,uncovered

@njit
def numba_greedy(adj_list,start,end,budget,node_weights):
    
    N= start.shape[0]
    solution = np.ones(N)*-1
    gains = get_gains(adj_list,start,end)
    uncovered = np.ones(N)
    can_select = np.ones(N)


    constraint =0
    idx = 0
    for i in range(N):

        max_gain_ratio = 0
        selected_element = -1

        for i in range(N):
            if can_select[i] == 1:
                if gains[i]/node_weights[i] > max_gain_ratio:
                    max_gain_ratio = gains[i]/node_weights[i]
                    selected_element = i


        if max_gain_ratio == 0:
            break

        if node_weights[selected_element]+constraint <= budget:
            solution[idx] = selected_element
            gain_adjustment(adj_list,start,end,gains,selected_element,uncovered)
            constraint += node_weights[selected_element]
            idx +=1

        can_select[selected_element] = 0
        
    return solution

# @njit
# def numba_greedy(adj_list,start,end,budget):

#     solution = np.ones(budget)*-1
#     gains = get_gains(adj_list,start,end)
#     N= start.shape[0]
#     uncovered = np.ones(N)
#     for idx in range(budget):
#         selected_element = np.argmax(gains)

#         if gains[selected_element]==0:
#             break
#         else:
#             solution[idx] = selected_element
#             gains,uncovered=gain_adjustment(adj_list,start,end,gains,selected_element,uncovered)
            

#     return solution




    


 


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument( "--dataset", type=str, default='Facebook', help="Name of the dataset to be used (default: 'Facebook')" )
    parser.add_argument( "--budgets", nargs='+', type=int, help="Budgets" )
    
    parser.add_argument("--cost_model",type= str, default= 'random', help = 'model of node weights')
  
    args = parser.parse_args()

    file_path=f'../../data/test/{args.dataset}'

    # file_path=f'../../data/s/{args.dataset}'
    graph=load_from_pickle(file_path)
    graph,forward_transformation_dic = relabel_graph(graph=graph,return_forward_transformation_dic=True)
    node_weights_dic = load_from_pickle(f'../../data/test/{args.dataset}_weights_{args.cost_model}')

    adj_list,start,end = from_networkx(graph)

    # node_weights= np.array(node_weights.values())
    # print( numba_greedy(adj_list,start,end,args.budgets[0],node_weights))
    node_weights = np.zeros(graph.number_of_nodes())

    for node in node_weights_dic:
        node_weights[forward_transformation_dic[node]]=node_weights_dic[node]
    # load_graph_file_path=f'../../data/snap_dataset/{args.dataset}.txt'
    # graph=nx.read_edgelist(f'../../data/snap_dataset/{args.dataset}.txt', create_using=nx.Graph(), nodetype=int)

    
    # df=defaultdict(list)

    # for budget in args.budgets:

    #     solution,_= modified_greedy(graph=graph,budget=budget,node_weights=node_weights)

    #     subgraph =make_subgraph(graph,solution)
    #     Pv=1-subgraph.number_of_nodes()/graph.number_of_nodes()
    #     Pe=1-subgraph.number_of_edges()/graph.number_of_edges()

    #     cover=calculate_cover(graph,solution)

    #     df['Budget'].append(budget)
    #     df['Pv'].append(Pv)
    #     df['Pe'].append(Pe)
    #     df['Objective Value'].append(cover)
    #     df['Objective Value (Ratio)'].append(cover/graph.number_of_nodes())
    #     df['Solution'].append(solution)
    #     df['Cost'].append([node_weights[node] for node in solution])
        
        
    # df=pd.DataFrame(df)
    # print(df)
    # save_folder=f'data/{args.dataset}'
    # file_path=os.path.join(save_folder,'Greedy')
    # os.makedirs(save_folder,exist_ok=True)
    # save_to_pickle(df,file_path)








