from argparse import ArgumentParser
from utils import *
# import pandas as pd
from collections import defaultdict
import numpy as np
import os
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB


def LP (dataset,budgets,node_weights = None):

    graph= load_from_pickle(f'../../data/test/{dataset}')
    graph,transformation, reverse_transformation = relabel_graph(graph=graph,return_reverse_transformation_dic= True, return_forward_transformation_dic= True)
    regions, population =gp.multidict({node: 1 for node in graph.nodes()})
    if node_weights is None:
       node_weights =  {node: 1 for node in graph.nodes()}


    multi_dict = {}
    for node in graph.nodes():
        covers = {neighbor for neighbor in graph.neighbors(node)}
        covers.add(node)
        multi_dict[node] = [covers , node_weights[node]]

    sites, coverage, cost = gp.multidict(multi_dict)
    # print(sites)
    # print(coverage)
    # print(cost)

    for budget in budgets:
        m = gp.Model("max_cover")

        build = m.addVars(len(sites), vtype=GRB.BINARY, name="Build")
        is_covered = m.addVars(len(regions), vtype=GRB.BINARY, name="Is_covered")

        m.addConstrs((gp.quicksum(build[t] for t in sites if r in coverage[t]) >= is_covered[r]
                                for r in regions), name="Build2cover")
        m.addConstr(build.prod(cost) <= budget, name="budget")

        m.setObjective(is_covered.prod(population), GRB.MAXIMIZE)

        m.optimize()

        solutions = []
        # print(build.keys())
        for node in build.keys():
            if (abs(build[node].x) > 1e-6):
                solutions.append(node)
                # print(f"\n Build a cell tower at location Tower {tower}.")
        print(solutions)
        print([graph.degree(node) for node in solutions])
        print('Cover:',calculate_cover(graph,solutions))
    

    # mapping to LP





if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument( "--dataset", type=str, default='Facebook', help="Name of the dataset to be used for training (default: 'Facebook')" )
    parser.add_argument( "--budgets", nargs='+', type=int, help="Budgets" )

    args = parser.parse_args()

    LP (dataset=args.dataset,budgets=args.budgets)