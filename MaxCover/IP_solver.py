import gurobipy as gp
from gurobipy import GRB
from argparse import ArgumentParser
from utils import *
import numpy as np


def gurobi_solver(graph:nx.Graph,budget:int,node_weights:dict,max_time = None,max_threads = None):

    graph,forward_mapping,reverse_mapping = relabel_graph(graph)
    
    

    node_weights = {forward_mapping[node]:node_weights[node] for node in graph.nodes()}

    regions, population = gp.multidict({node:1 for node in graph.nodes()})
    

    temp_dict = {}

    for node in graph.nodes():

        neighbors = set(graph.neighbors(node))
        neighbors.add(node)
        temp_dict[node] = [neighbors,node_weights[node]]
    sites, coverage, cost = gp.multidict(temp_dict)

    m = gp.Model("cell_tower")
    if max_time is None:
        m.setParam('TimeLimit', 600)

    if max_threads:
        m.setParam('Threads', max_threads)

    build = m.addVars(len(sites), vtype=GRB.BINARY, name="Build")
    is_covered = m.addVars(len(regions), vtype=GRB.BINARY, name="Is_covered")

    m.addConstrs((gp.quicksum(build[t] for t in sites if r in coverage[t]) >= is_covered[r]
                            for r in regions), name="Build2cover")
    m.addConstr(build.prod(cost) <= budget, name="budget")

    m.setObjective(is_covered.prod(population), GRB.MAXIMIZE)

    m.optimize() 

    # display optimal values of decision variables
    solution = []
    for tower in build.keys():
        if (abs(build[tower].x) > 1e-6):
            solution.append(reverse_mapping[tower])
            
    return solution, m.ObjVal


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default='Facebook', help="Name of the dataset to be used (default: 'Facebook')")
    parser.add_argument("--budget", type=int,default=10, help="Budgets")
    parser.add_argument("--cost_model",type= str, default= 'random', help = 'model of node weights')


    args = parser.parse_args()
    graph = nx.read_edgelist(f'../../data/snap_dataset/{args.dataset}.txt', create_using=nx.Graph(), nodetype=int)

    if args.cost_model == 'uniform':
        node_weights = {node:1 for node in graph.nodes()}

    elif args.cost_model == 'degree':
        # alpha = 1/20
        alpha = 1/20
        out_degrees = {node: graph.degree(node) for node in graph.nodes()}
        out_degree_max = np.max(list(out_degrees.values()))
        out_degree_min = np.min(list(out_degrees.values()))
        node_weights = {node: (out_degrees[node] - out_degree_min + alpha) / (out_degree_max - out_degree_min) for node in graph.nodes()}

    else:
        raise NotImplementedError('Unknown model')
    

    solution =gurobi_solver(graph=graph,budget=args.budget,node_weights=node_weights)

    print(solution)