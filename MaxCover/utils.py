import pickle
import networkx as nx
import os







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


