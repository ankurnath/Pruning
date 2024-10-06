
from utils import *

from greedy import greedy,gain_adjustment,get_gains

from knapsack_numba_greedy import knapsack_numba_greedy
import matplotlib.pyplot as plt
from helper_functions import calculate_obj


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument( "--dataset", type=str, default='Facebook', help="Name of the dataset to be used (default: 'Facebook')" )
    parser.add_argument( "--budget", type=int, default=100, help='Budget' )
    parser.add_argument("--cost_model",type=str,default='aistats',help='model of node weights')
    parser.add_argument( "--size", type=int, default=750, help='size' )
    args = parser.parse_args()

    cost_model = args.cost_model
    load_graph_file_path=f'../../data/snap_dataset/{args.dataset}.txt'
    graph = load_graph(load_graph_file_path)
    node_weights = generate_node_weights(graph=graph,cost_model=cost_model)
    max_weight_node = max(node_weights,key=node_weights.get)

    sprint(node_weights[max_weight_node])


    

