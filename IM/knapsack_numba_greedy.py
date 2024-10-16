from numba import njit
import numpy as np
def process_rr_sets(graph, gains_dict, node_rr_set, RR):
    N = graph.number_of_nodes()
    gains = np.zeros(N)
    node_rr_set_start_index = np.zeros(N,dtype=np.int32)
    node_rr_set_end_index = np.zeros(N,dtype=np.int32)
    node_rr_set_flat = []

    for key in gains_dict:
        gains[key] = gains_dict[key]
        node_rr_set_start_index[key] = len(node_rr_set_flat) - 1
        node_rr_set_flat += node_rr_set[key]
        node_rr_set_end_index[key] = len(node_rr_set_flat) - 1

    node_rr_set_flat = np.array(node_rr_set_flat)

    rr_start_index = np.zeros(len(RR),dtype=np.int32)
    rr_end_index = np.zeros(len(RR),dtype=np.int32)
    RR_flat = []

    for i, rr in enumerate(RR):
        rr_start_index[i] = len(RR_flat) - 1
        RR_flat += rr
        rr_end_index[i] = len(RR_flat) - 1

    RR_flat = np.array(RR_flat)

    return gains, node_rr_set_start_index, node_rr_set_end_index, node_rr_set_flat, rr_start_index, rr_end_index, RR_flat




@njit 
def numba_gain_adjustment(node_rr_set_flat, node_rr_set_start_index, node_rr_set_end_index,
                     RR_flat, rr_start_index, rr_end_index,gains,selected_element,covered_rr_set):
    
    # Loop through the relevant portion of node_rr_set_flat for the selected element
    for index in node_rr_set_flat[node_rr_set_start_index[selected_element]:node_rr_set_end_index[selected_element]]:
        if covered_rr_set[index] == 1:  # If the RR set is not yet covered
            covered_rr_set[index] = 0  # Mark the RR set as covered
            
            # Update the gains for nodes in the RR set
            for rr_node in RR_flat[rr_start_index[index]:rr_end_index[index]]:
                gains[rr_node] -= 1
    


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

@njit 
def knapsack_numba_greedy(gains,node_weights,budget,node_rr_set_start_index, node_rr_set_end_index, 
                          node_rr_set_flat, rr_start_index, rr_end_index, RR_flat,mask):
    

    N = gains.shape[0]
    covered_rr_set = np.zeros(rr_start_index.shape[0])
    solution = np.zeros(N)
    objective_value = 0 
    constraint = 0 
    for _ in range(N):

        selected_element = select_element(gains,node_weights,mask)

        if selected_element == -1:
            break

        elif constraint+node_weights[selected_element] <= budget:
            constraint += node_weights[selected_element]
            objective_value+= gains[selected_element]
            solution[selected_element] = 1
            numba_gain_adjustment(node_rr_set_flat, node_rr_set_start_index, node_rr_set_end_index,
                     RR_flat, rr_start_index, rr_end_index,gains,selected_element,covered_rr_set)
            

    return objective_value,solution
            


def knapsack_greedy(graph,budget,node_weights,gains,node_rr_set,RR,num_rr,ground_set=None):

    # first step: Create the mask

    N = graph.number_of_nodes()
    mask = np.ones(N)

    node_weights = np.array([node_weights[node] for node in range(N)])

    if ground_set is not None:
        mask = np.zeros(N)
        for node in ground_set:
            mask[node] = 1

    
    size_of_ground_set = np.sum(mask)

    number_of_queries = size_of_ground_set * (size_of_ground_set - 1)/2

    gains, node_rr_set_start_index, node_rr_set_end_index, node_rr_set_flat, rr_start_index, rr_end_index, RR_flat =process_rr_sets(graph=graph, gains_dict=gains, node_rr_set=node_rr_set, RR=RR)

    objective_value,solution = knapsack_numba_greedy(gains=gains,
                                                     node_weights=node_weights,
                                                     budget=budget,
                                                     node_rr_set_start_index=node_rr_set_start_index,
                                                     node_rr_set_end_index=node_rr_set_end_index,
                                                     node_rr_set_flat = node_rr_set_flat,
                                                     rr_start_index=rr_start_index,
                                                     rr_end_index = rr_end_index,
                                                     RR_flat=RR_flat,
                                                     mask = mask)
    


    solution = np.where(solution == 1)[0].tolist()


    return objective_value,solution,number_of_queries

        
# from utils import *   
# from greedy import get_gains
# dataset = 'Facebook'
# cost_model = 'aistats'
# num_rr = 100000
# budget  = 100

# load_graph_file_path=f'../../data/snap_dataset/{dataset}.txt'
# graph = load_graph(load_graph_file_path)
# node_weights = generate_node_weights(graph=graph,cost_model=cost_model)

# gains,node_rr_set,RR = get_gains(graph,num_rr)

# objective_value,solution,number_of_queries=knapsack_greedy(graph,budget,node_weights,gains,node_rr_set,RR,ground_set=None)

# print(objective_value,solution,number_of_queries)