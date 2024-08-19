import networkx as nx
def calculate_obj(graph: nx.Graph, solution):

    covered_elements=set()
    for node in solution:
        covered_elements.add(node)
        for neighbour in graph.neighbors(node):
            covered_elements.add(neighbour)
    
    return len(covered_elements)