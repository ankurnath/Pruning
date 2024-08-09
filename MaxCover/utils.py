import pickle
import networkx as nx
import os
from smartprint import smartprint as sprint
import time






def calculate_cover(graph: nx.Graph, selected_nodes):

    covered_elements=set()

    for node in selected_nodes:
        covered_elements.add(node)
        for neighbour in graph.neighbors(node):
            covered_elements.add(neighbour)
    
    return len(covered_elements)

def make_subgraph(graph, nodes):
    assert type(graph) == nx.Graph
    subgraph = nx.Graph()
    edges_to_add = []
    for node in nodes:
        edges_to_add += [(u, v) for u, v in list(graph.edges(node))]
    subgraph.add_edges_from(edges_to_add)
    return subgraph

def load_from_pickle(file_path):
    """
    Load data from a pickle file.

    Parameters:
    - file_path: The path to the pickle file.

    Returns:
    - loaded_data: The loaded data.
    """
    with open(file_path, 'rb') as file:
        loaded_data = pickle.load(file)
    print(f'Data has been loaded from {file_path}')
    return loaded_data


def save_to_pickle(data, file_path):
    """
    Save data to a pickle file.

    Parameters:
    - data: The data to be saved.
    - file_path: The path to the pickle file.
    """
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)
    print(f'Data has been saved to {file_path}')

import networkx as nx

def relabel_graph(graph: nx.Graph):
    """
    Relabel the nodes of the input graph to have consecutive integer labels starting from 0.

    Parameters:
    graph (nx.Graph): The input graph to be relabeled.

    Returns:
    tuple: A tuple containing the relabeled graph, a forward mapping dictionary, 
           and a reverse mapping dictionary.
           - relabeled_graph (nx.Graph): The graph with nodes relabeled to consecutive integers.
           - forward_mapping (dict): A dictionary mapping original node labels to new integer labels.
           - reverse_mapping (dict): A dictionary mapping new integer labels back to the original node labels.
    """
    forward_mapping = dict(zip(graph.nodes(), range(graph.number_of_nodes())))
    reverse_mapping = dict(zip(range(graph.number_of_nodes()), graph.nodes()))
    graph = nx.relabel_nodes(graph, forward_mapping)

    return graph, forward_mapping, reverse_mapping


    # if return_reverse_transformation_dic and return_forward_transformation_dic:
    #     return nx.relabel_nodes(graph, transformation), transformation, reverse_transformation

    # elif return_forward_transformation_dic:
    #     return nx.relabel_nodes(graph, transformation), transformation

    # elif return_reverse_transformation_dic:
    #     return nx.relabel_nodes(graph, transformation), reverse_transformation
    # nodes = graph.nodes()
    # n = graph.number_of_nodes()
    # desired_labels = set([i for i in range(n)])
    # already_labeled = set([int(node) for node in nodes if node < n])
    # desired_labels = desired_labels - already_labeled
    # transformation = {}
    # reverse_transformation = {}
    # for node in nodes:
    #     if node >= graph.number_of_nodes():
    #         transformation[node] = desired_labels.pop()
    #         reverse_transformation[transformation[node]] = node

    # if return_reverse_transformation_dic and return_forward_transformation_dic:
    #     return nx.relabel_nodes(graph, transformation), transformation, reverse_transformation

    # elif return_forward_transformation_dic:
    #     return nx.relabel_nodes(graph, transformation), transformation

    # elif return_reverse_transformation_dic:
    #     return nx.relabel_nodes(graph, transformation), reverse_transformation
