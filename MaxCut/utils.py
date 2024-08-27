import os
import numpy as np
import networkx as nx
from smartprint import smartprint as sprint
import time
from tqdm import tqdm
import pandas as pd
import pickle
from argparse import ArgumentParser
from collections import defaultdict
import matplotlib.pyplot as plt
import random
from numba import njit


def generate_node_weights(graph,cost_model):

    if cost_model == 'uniform':
        node_weights = {node:1 for node in graph.nodes()}

    if cost_model == 'random':
        node_weights = {node:random.random() for node in graph.nodes()}

    elif cost_model == 'degree':
        
        alpha = 1/20
        out_degrees = {node: graph.degree(node) for node in graph.nodes()}
        out_degree_max = np.max(list(out_degrees.values()))
        out_degree_min = np.min(list(out_degrees.values()))
        node_weights = {node: (out_degrees[node] - out_degree_min + alpha) / (out_degree_max - out_degree_min) for node in graph.nodes()}

    else:
        raise NotImplementedError('Unknown model')
    
    return node_weights



def make_subgraph(graph, nodes):
    assert type(graph) == nx.Graph
    subgraph = nx.Graph()
    edges_to_add = []
    for node in nodes:
        edges_to_add += [(u, v) for u, v in list(graph.edges(node))]
    subgraph.add_edges_from(edges_to_add)
    return subgraph


def load_graph(file_path):

    try:
        graph = nx.read_edgelist(file_path, create_using=nx.Graph(), nodetype=int)

    except:
        f = open(file_path, mode="r")
        lines = f.readlines()
        edges = []

        for line in lines:
            line = line.split()
            if line[0].isdigit():
                edges.append([int(line[0]), int(line[1])])
        graph = nx.Graph()
        graph.add_edges_from(edges)


    graph.remove_edges_from(list(nx.selfloop_edges(graph)))
    return graph


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