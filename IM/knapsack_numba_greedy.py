from numba import njit
import numpy as np
def process_rr_sets(graph, gains_dict, node_rr_set, RR):
    N = graph.number_of_nodes()
    gains = np.zeros(N)
    node_rr_set_start_index = np.zeros(N)
    node_rr_set_end_index = np.zeros(N)
    node_rr_set_flat = []

    for key in gains_dict:
        gains[key] = gains_dict[key]
        node_rr_set_start_index[key] = len(node_rr_set_flat) - 1
        node_rr_set_flat += node_rr_set[key]
        node_rr_set_end_index[key] = len(node_rr_set_flat) - 1

    node_rr_set_flat = np.array(node_rr_set_flat)

    rr_start_index = np.zeros(len(RR))
    rr_end_index = np.zeros(len(RR))
    RR_flat = []

    for i, rr in enumerate(RR):
        rr_start_index[i] = len(RR_flat) - 1
        RR_flat += rr
        rr_end_index[i] = len(RR_flat) - 1

    RR_flat = np.array(RR_flat)

    return gains, node_rr_set_start_index, node_rr_set_end_index, node_rr_set_flat, rr_start_index, rr_end_index, RR_flat




@njit
def gain_adjustment(node_rr_set_flat, node_rr_set_start_index, node_rr_set_end_index,
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