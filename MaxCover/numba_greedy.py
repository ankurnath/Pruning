from numba import njit
from argparse import ArgumentParser
import networkx as nx

from utils import *
import numpy as np
import time
from tqdm import tqdm


def flatten_graph(graph:nx.Graph):
    flat_adj_matrix = []
    
    n = graph.number_of_nodes()
    start = [0 for _ in range(n)]
    end = [0 for _ in range(n)]
    adj_list_dict = nx.to_dict_of_lists(graph)

    

    for node, neighbors in adj_list_dict.items():
        start[node] = len(flat_adj_matrix)
        end[node] = start[node] + len(neighbors)
        flat_adj_matrix += neighbors
    return np.array(flat_adj_matrix), np.array(start), np.array(end)

@njit
def get_gains(adj_list,start,end):

    N = len(start)
    gains = np.zeros(N)

    for node in range(N):
        gains[node]=end[node]-start[node]+1

    return gains


@njit
def gain_adjustment(adj_list,start,end,gains,selected_element,uncovered):


    if uncovered[selected_element] ==1 :
        gains[selected_element]-=1
        uncovered[selected_element]= 0
        for neighbor in adj_list[start[selected_element]:end[selected_element]]:
            gains[neighbor]-=1

    for neighbor in adj_list[start[selected_element]:end[selected_element]]:
        if uncovered[neighbor] == 1:
            uncovered[neighbor]= 0
            gains[neighbor]-=1
            for neighbor_of_neighbor in adj_list[start[neighbor]:end[neighbor]]:
                gains[neighbor_of_neighbor ]-=1


    assert gains[selected_element] == 0


@njit
def select_element(gains,node_weights,mask):

    N = len(gains)

    max_gain_ratio = 0
    selected_element = -1
    for node in range(N):
        if mask[node] == 1 and gains[node]/node_weights[node] >= max_gain_ratio:
            max_gain_ratio = gains[node]/node_weights[node]
            selected_element = node

    mask [selected_element] = 0
    return selected_element


# @njit
# def get_solution(adj_list,start,end,node_weights,budget):

#     N= len(start)
#     number_queries = 0
#     gains = get_gains(adj_list=adj_list,start=start,end=end)
#     uncovered = np.ones(N)
#     mask = np.ones(N)
#     solution = np.zeros(N)

#     constraint = 0
#     for i in range(N):
        
#         selected_element = select_element(gains,node_weights,mask)
#         # selected_element = np.argmax(gains)

#         if node_weights[selected_element] + constraint <= budget:
#             constraint += node_weights[selected_element]
#             gain_adjustment(adj_list=adj_list,start=start,end=end,
#                         gains=gains,selected_element=selected_element,uncovered=uncovered)
#             solution[selected_element] = 1
        
#         # print(gains[selected_element])
    
#     return solution


def numba_greedy(graph:nx.Graph,budget:int,node_weights:dict,ground_set=None):
    # forw
    graph, forward_mapping, reverse_mapping = relabel_graph(graph=graph)
    
    
    node_weights= np.array([node_weights[reverse_mapping[node]] for node in reverse_mapping])

    adj_list,start,end=flatten_graph(graph=graph)

    start_time = time.time()

    # solution = get_solution(adj_list,start,end,node_weights,budget)
    # solution = np.nonzero(array)[0]
    N = graph.number_of_nodes()
    
    gains = get_gains(adj_list=adj_list,start=start,end=end)
    uncovered = np.ones(N)
    number_of_queries = 0
    if ground_set is None:
        ground_set_size = N

        mask = np.ones(N)
    else:
        ground_set_size = len(ground_set)
        mask = np.zeros(N)
        for element in ground_set:
            mask[forward_mapping[element]] = 1

    # solution = np.zeros(N)
    
    # max singleton
            
    
    # Initialize variables
    max_gain = 0
    max_node = None

    # Iterate through each node
    for node in range(N):
        # Check if the node is in the mask, within budget, and has a higher gain than the current max
        if mask[node] == 1 and node_weights[node] <= budget:
            number_of_queries += 1
            if  gains[node] > max_gain:
                max_gain = gains[node]
                max_node = node
            

    solution = []

    start_time = time.time()

    constraint = 0
    for i in tqdm(range(N)):
        number_of_queries += (ground_set_size- i)
        selected_element = select_element(gains,node_weights,mask)
        # selected_element = np.argmax(gains)

        if selected_element != -1 and gains[selected_element] !=0 and node_weights[selected_element] + constraint <= budget:
            constraint += node_weights[selected_element]
            gain_adjustment(adj_list=adj_list,start=start,end=end,
                        gains=gains,selected_element=selected_element,uncovered=uncovered)
            # solution[selected_element] = 1
            solution.append(selected_element)

    end_time = time.time()

    print("Elapsed time:",round(end_time-start_time,4))

    # print('Solution',solution)
    # print('Degree:',[graph.degree(node) for node in solution])
    # print(number_of_queries)
    if calculate_cover(graph,solution)>=calculate_cover(graph,[max_node]):
        # print('Cover:',calculate_cover(graph,solution))
        return [reverse_mapping[node] for node in solution],number_of_queries
    else:
        return [reverse_mapping[max_node]],number_of_queries


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default='Facebook', help="Name of the dataset to be used (default: 'Facebook')")
    parser.add_argument("--budget", type=int,default=10, help="Budgets")
    parser.add_argument("--cost_model",type= str, default= 'degree', help = 'model of node weights')
    


    args = parser.parse_args()

    graph = nx.read_edgelist(f'../../data/snap_dataset/{args.dataset}.txt', create_using=nx.Graph(), nodetype=int)
    
    # node_weights = {node:1 for node in graph.nodes()}

    if args.cost_model == 'uniform':
        node_weights = {node:1 for node in graph.nodes()}

    elif args.cost_model == 'degree':
        # alpha = 1/20
        alpha = 1
        out_degrees = {node: graph.degree(node) for node in graph.nodes()}
        out_degree_max = np.max(list(out_degrees.values()))
        out_degree_min = np.min(list(out_degrees.values()))
        node_weights = {node: (out_degrees[node] - out_degree_min + alpha) / (out_degree_max - out_degree_min) for node in graph.nodes()}

    else:
        raise NotImplementedError('Unknown model')



    numba_greedy(graph=graph,budget=args.budget,node_weights=node_weights)
    


