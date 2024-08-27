import networkx as nx
def calculate_obj(graph: nx.Graph, solution):

    cut = 0

    solution = set(solution)

    for node in solution:
        for neighbor in graph.neighbors(node):
            if neighbor not in solution:
                cut+=1
    return cut


