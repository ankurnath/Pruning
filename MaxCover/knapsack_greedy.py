
from utils import *
from greedy import get_gains, gain_adjustment
from helper_functions import *


def knapsack_greedy(graph,budget,ground_set=None , node_weights= None):
    
    number_of_queries=0

    gains=get_gains(graph,ground_set)

    # get max singleton

    max_singleton = None

    max_singleton_gain = 0

    number_of_queries += len(gains)
    for element in gains:
        if node_weights[element]<= budget and gains[element] > max_singleton_gain:
            max_singleton = element
            max_singleton_gain = gains [element]


    solution=[]
    uncovered=defaultdict(lambda: True)

    constraint = 0

    N = len (gains)

    


    while gains:
        number_of_queries+= len(gains)
        max_gain_ratio = 0

        selected_element = None

        for element in gains:
            if gains[element]/node_weights[element]> max_gain_ratio:
                max_gain_ratio = gains[element]/node_weights[element]
                selected_element = element

        if max_gain_ratio == 0 :

            break

        if node_weights[selected_element]+constraint <= budget:

            solution.append(selected_element)
            gain_adjustment(graph,gains,selected_element,uncovered)
            constraint += node_weights[selected_element]


        if constraint == budget:
            break

        gains.pop(selected_element)

    # print('Number of queries:',number_of_queries)
        
    if calculate_obj(graph,solution)< max_singleton_gain :
        solution = [ max_singleton ]

    return solution,number_of_queries

        


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument( "--dataset", type=str, default='Facebook', help="Name of the dataset to be used (default: 'Facebook')" )
    parser.add_argument( "--budgets", nargs='+', type=int, help="Budgets" )
    parser.add_argument("--cost_model",type= str, default= 'random', help = 'model of node weights')
  
    args = parser.parse_args()

    file_path=f'../../data/test/{args.dataset}'
    graph=load_from_pickle(file_path)

    cost_model = args.cost_model
    budget = args.budget
    node_weights = generate_node_weights(graph=graph,cost_model=cost_model)


    knapsack_greedy(graph = graph,budget = budget,node_weights=node_weights)









