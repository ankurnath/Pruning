import pickle
import networkx as nx
import os
import random
import multiprocessing as mp
import time
import math
import networkx as nx
from collections import defaultdict
from collections import deque
from multiprocessing import Process
import numpy as np
from multiprocessing.pool import Pool

NUM_PROCESSORS = mp.cpu_count()

class Worker(mp.Process):
    def __init__(self, inQ, outQ, node_num, model, graph_):
        super(Worker, self).__init__(target=self.start)
        self.inQ = inQ
        self.outQ = outQ
        self.R = []
        self.count = 0
        # self.node_num = node_num
        self.model = model
        self.graph = graph_
        self.nodes= list(graph_.network.keys())

    def run(self):

        while True:
            theta = self.inQ.get()
            while self.count < theta:
                # v = random.randint(1, self.node_num-1)
                v=random.choice(self.nodes)

                rr = generate_rr(v, self.model, self.graph)
                self.R.append(rr)
                self.count += 1
            self.count = 0
            self.outQ.put(self.R)
            self.R = []

def create_worker(num, worker, node_num, model, graph_):
    for i in range(num):
        worker.append(Worker(mp.Queue(), mp.Queue(), node_num, model, graph_))
        worker[i].start()


def finish_worker(worker):
    for w in worker:
        w.terminate()

def sampling(epsoid, l, node_num, seed_size, worker, graph_, model):
    R = []
    LB = 1
    n = node_num
    k = seed_size
    epsoid_p = epsoid * math.sqrt(2)
    worker_num = NUM_PROCESSORS
    create_worker(worker_num, worker, node_num, model, graph_)
    for i in range(1, int(math.log2(n - 1)) + 1):
        # s = time.time()
        x = n / (math.pow(2, i))
        lambda_p = ((2 + 2 * epsoid_p / 3) * (logcnk(n, k) + l * math.log(n) + math.log(math.log2(n))) * n) / pow(
            epsoid_p, 2)
        theta = lambda_p / x

        # print(f'Creating new {theta-len(R)} RR sets')
        for ii in range(worker_num):
            worker[ii].inQ.put((theta - len(R)) / worker_num)
        for w in worker:
            R_list = w.outQ.get()
            R += R_list
        # finish_worker()
        # worker = []
        # end = time.time()
        # print('time to find rr', end - s)
        start = time.time()
        Si, f, my_variable,_= node_selection(R, k, node_num)
        end = time.time()
        # print('node selection time', time.time() - start)
        # f = F(R,Si)
        if n * f >= (1 + epsoid_p) * x:
            LB = n * f / (1 + epsoid_p)
            break
    # finish_worker()
    alpha = math.sqrt(l * math.log(n) + math.log(2))
    beta = math.sqrt((1 - 1 / math.e) * (logcnk(n, k) + l * math.log(n) + math.log(2)))
    lambda_aster = 2 * n * pow(((1 - 1 / math.e) * alpha + beta), 2) * pow(epsoid, -2)
    theta = lambda_aster / LB
    length_r = len(R)
    diff = theta - length_r
    _start = time.time()
    if diff > 0:
        for ii in range(worker_num):
            worker[ii].inQ.put(diff / worker_num)
        for w in worker:
            R_list = w.outQ.get()
            R += R_list
    '''

    while length_r <= theta:
        v = random.randint(1, n)
        rr = generate_rr(v)
        R.append(rr)
        length_r += 1
    '''
    _end = time.time()
    finish_worker(worker)
    print('Number of RR sets:',len(R))
    return R



def generate_rr(v, model, graph):
    if model == 'IC':
        return generate_rr_ic(v, graph)
    elif model == 'LT':
        return generate_rr_lt(v, graph)
    

def node_selection(R,k,node_num=None):
    Sk=set()
    list1=[]
    rr_degree=defaultdict(int)
    node_rr_set = defaultdict(list)
    matched_count = 0

    covered_rr_set=set()
    for j,rr in enumerate(R):
        for rr_node in rr:
            rr_degree[rr_node]+=1
            node_rr_set[rr_node].append(j)
    
    merginal_gains=[]
    for i in range(k):
        max_point=max(rr_degree,key=rr_degree.get)
        Sk.add(max_point)
        list1.append(max_point)
        matched_count +=rr_degree[max_point]
        merginal_gains.append(rr_degree[max_point]/len(R))

        for index in node_rr_set[max_point]:
            if index not in covered_rr_set:
                covered_rr_set.add(index)
                for rr_node in R[index]:
                    rr_degree[rr_node]-=1

    return Sk, matched_count / len(R), list1,merginal_gains


def generate_rr_ic(node, graph):
    activity_set = list()
    activity_set.append(node)
    # activity_nodes = list()
    activity_nodes = set()
    # activity_nodes.append(node)
    activity_nodes.add(node)
    while activity_set:
        new_activity_set = list()
        for seed in activity_set:
            for node, weight in graph.get_neighbors(seed):
                if node not in activity_nodes:
                    if random.random() < weight:
                        # activity_nodes.append(node)
                        activity_nodes.add(node)
                        new_activity_set.append(node)
        activity_set = new_activity_set
    return activity_nodes


def generate_rr_lt(node, graph):
    # calculate reverse reachable set using LT model
    # activity_set = list()
    activity_nodes = list()
    # activity_set.append(node)
    activity_nodes.append(node)
    activity_set = node

    while activity_set != -1:
        new_activity_set = -1

        neighbors = graph.get_neighbors(activity_set)
        if len(neighbors) == 0:
            break
        candidate = random.sample(neighbors, 1)[0][0]
        if candidate not in activity_nodes:
            activity_nodes.append(candidate)
            # new_activity_set.append(candidate)
            new_activity_set = candidate
        activity_set = new_activity_set
    return activity_nodes

def logcnk(n, k):
    res = 0
    for i in range(n - k + 1, n + 1):
        res += math.log(i)
    for i in range(1, k + 1):
        res -= math.log(i)
    return res


def get_graph(network):
    """
    Takes in either filepath to graph object or a networkx object
    """
    if type(network) == str:
        graph_ = Graph()
        data_lines = open(network, 'r').readlines()
        node_num = int(float(data_lines[0].split()[0]))
        # edge_num = int(float(data_lines[0].split()[1]))

        for data_line in data_lines[1:]:
            start, end, weight = data_line.split()
            graph_.add_edge(int(float(start)), int(float(end)), float(weight))
            # pGraph.add_edge(int(float(start)), int(float(end)), float(weight))
        return graph_, node_num

    elif type(network) == nx.Graph or type(network) == nx.DiGraph:
        graph_ = Graph()
        node_num = network.number_of_nodes()

        for u, v in network.edges():
            weight = network[u][v]['weight']
            if type(weight) == dict:
                weight = weight['weight']
            graph_.add_edge(int(u), int(v), weight)
            if type(network) == nx.Graph:
                graph_.add_edge(int(v), int(u), network[v][u]['weight'])

        return graph_, node_num
    
class Graph:
    """
        graph data structure to store the network
    :return:
    """
    def __init__(self):
        self.network = dict()

    def add_node(self, node):
        if node not in self.network:
            self.network[node] = dict()

    def add_edge(self, s, e, w):
        """
        :param s: start node
        :param e: end node
        :param w: weight
        """
        Graph.add_node(self, s)
        Graph.add_node(self, e)
        # add inverse edge
        self.network[e][s] = w

    def get_out_degree(self, source):
        return len(self.network[source])

    def get_neighbors(self, source):
        if source in self.network:
            return self.network[source].items()
        else:
            return []

    def get_neighbors_keys(self, source):
        if source in self.network:
            return self.network[source].keys()
        else:
            return []


def make_subgraph(graph, nodes):
    assert type(graph) == nx.DiGraph
    subgraph = nx.DiGraph()
    edges_to_add = []
    for node in nodes:
        # edges_to_add += [(u, v, w) for u, v, w in list(graph.out_edges(node, data=True)) + list(graph.in_edges(node, data=True))]
        edges_to_add += [(u, v, w) for u, v, w in list(graph.out_edges(node, data=True))]
    subgraph.add_weighted_edges_from(edges_to_add)
    return subgraph
# def make_subgraph(graph, nodes):
#     assert type(graph) == nx.Graph or type(graph) == nx.DiGraph
#     subgraph = nx.DiGraph()
#     edges_to_add = []
#     for node in nodes:
#         edges_to_add += [(u, v) for u, v in list(graph.edges(node))]
#     subgraph.add_edges_from(edges_to_add)
#     return subgraph

def load_from_pickle(file_path,quiet = False):
    """
    Load data from a pickle file.

    Parameters:
    - file_path: The path to the pickle file.

    Returns:
    - loaded_data: The loaded data.
    """
    with open(file_path, 'rb') as file:
        loaded_data = pickle.load(file)

    if not quiet:
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



def bfs(file_path, initial_nodes):
    subgraph = load_from_pickle(file_path=file_path,quiet = True)
    activated_nodes = set(initial_nodes)
    queue = deque(initial_nodes)
    # subgraph_nodes = set(subgraph.nodes())

    while queue:
        node = queue.popleft()

        if node not in subgraph.nodes():
            continue

        for neighbor in subgraph.neighbors(node):
            if neighbor not in activated_nodes:
                activated_nodes.add(neighbor)
                queue.append(neighbor)

    return len(activated_nodes)


def calculate_spread(folder_path,solution):

    totol_spread = 0

    files = os.listdir (folder_path)

    # solution = set (solution)

    # file_paths = []

    # for file in files:
    #     # subgraph = load_from_pickle(os.path.join(folder_path,file),quit=True)
    #     file_paths.append(file)

    step_size = 1000
    for i in range(0,len(files),step_size):
        arguments=[]
        for _ in range(i,i+step_size):
            arguments.append ((os.path.join(folder_path,files[i]),solution))
            # process = Process(target=bfs, args=(os.path.join(folder_path,files[i]),))
            # processes.append(process)
            # process.start()
        with Pool() as pool:
            totol_spread += np.sum(pool.starmap(bfs,arguments))
        # best_cut=np.max(pool.starmap(tabu, arguments)) 

        # for process in processes:
        #     process.join()


    #     # activated_nodes = nx.descendants(subgraph,solution)

    #     activated_nodes = set (solution)
        
    #     queue = solution.copy()

    #     while queue:

    #         node = queue.pop(0)

    #         if node not in subgraph.nodes():
    #             continue

    #         for neighbor in subgraph.neighbors(node):
    #             if neighbor not in activated_nodes:
    #                 activated_nodes.add(neighbor)
    #                 queue.append(neighbor)
        
    #     totol_spread += len(activated_nodes)

    #     # activated_nodes = set (nx.descendants(subgraph,solution))

        
    #     # totol_spread += len(activated_nodes.union(set(solution)))

    return totol_spread/len(files)


