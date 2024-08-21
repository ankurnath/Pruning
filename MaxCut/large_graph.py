import networkx as nx
import pickle
import os
from collections import defaultdict


# dataset = 'friendster'
# file_path = f'../../data/snap_dataset/{dataset}.txt'

class Graph(object):

    def __init__(self,file_path) -> None:
        self.adj_list = defaultdict(list)
        file = open(file_path, 'r')
        self._number_of_edges = 0
        while True:
            line = file.readline()
            # print(line)
            if line[0].isdigit():
                break
        # print(line)
        u, v = map(int, line.split())
        self.adj_list[u].append(v)
        self.adj_list[v].append(u)
        
        while True:
            line = file.readline()
            if not line:
                break
            u, v = map(int, line.split())
            self.adj_list[u].append(v)
            self.adj_list[v].append(u)
        self._number_of_nodes = len(self.adj_list)

    
    def neighbors(self,node):
        return self.adj_list[node]
    
    def number_of_nodes(self):
        return self._number_of_nodes
    
    def number_of_edges(self):
        return self._number_of_edges
    

    def degree(self,node):
        return len(self.adj_list[node])
    
    def nodes(self):
        return self.adj_list.keys()
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
# from greedy import greedy
# from utils import calculate_cover


# sol,_=greedy(graph=graph,budget=5)
# # print(greedy(graph=graph,budget=5))
# print(calculate_cover(graph,sol))