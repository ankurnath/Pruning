import gurobipy as gp
from gurobipy import GRB
from argparse import ArgumentParser
from utils import *
import numpy as np


def gurobi_solver(graph:nx.Graph,budget:int,node_weights:dict,max_time = None,max_threads = None):

    graph,forward_mapping,reverse_mapping = relabel_graph(graph)

    node_weights = {forward_mapping[node]:node_weights[node] for node in forward_mapping}
        

    model = gp.Model("Max Cut solver")
    # if max_time is None:
    #     model.setParam('TimeLimit', 600)

    # else:
    #     model.setParam('TimeLimit',max_time)

    if max_time:
        model.setParam('TimeLimit',max_time)

    else:
        model.setParam('TimeLimit',3600)

    if max_threads:
        model.setParam('Threads', max_threads)

    vdict= model.addVars(graph.number_of_nodes(), vtype=GRB.BINARY, name="node")

    cut = [(vdict[i] + vdict[j] - 2*vdict[i]*vdict[j]) for i,j in graph.edges()]

    

    
    model.addConstr( vdict.prod(node_weights) <= budget, name="budget")

    model.setObjective(gp.quicksum(cut), GRB.MAXIMIZE)

    model.optimize() 

    # display optimal values of decision variables
    solution = []
    for node in vdict.keys():
        if (abs(vdict[node].x) > 1e-6):
            solution.append(reverse_mapping[node])
            
    return model.ObjVal,model.IterCount,solution


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default='Facebook', help="Name of the dataset to be used (default: 'Facebook')")
    parser.add_argument("--budget", type=int,default=10, help="Budgets")
    parser.add_argument("--cost_model",type= str, default= 'random', help = 'model of node weights')


    args = parser.parse_args()

    graph = load_graph(f'../../data/snap_dataset/{args.dataset}.txt')
    
    node_weights=generate_node_weights(graph=graph,cost_model=args.cost_model)
    

    solution = gurobi_solver(graph=graph,budget=args.budget,node_weights=node_weights)

    print(solution[0])