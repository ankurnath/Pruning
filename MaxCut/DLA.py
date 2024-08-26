from utils import *
from helper_functions import *

from greedy import *


def LA(graph:nx.Graph,gains:dict,node_weights:dict,budget:int):

    V_1 = [node for node in gains if node_weights[node]<=budget/2]

    max_singleton = None
    max_gain = 0

    for node in gains:
        if node_weights[node] <= budget and max_gain<= gains[node]:
            max_gain = gains[node]
            max_singleton = node 

    gains_X = gains.copy()
    gains_Y = gains.copy()

    spins_X = np.zeros(graph.number_of_nodes())
    spins_Y = np.zeros(graph.number_of_nodes())

    obj_X = 0
    obj_Y = 0

    X = []
    Y = []

    for node in tqdm(V_1):


        density_gain_X = gains_X[node]/node_weights[node]
        density_gain_Y = gains_Y[node]/node_weights[node]

        if density_gain_X  >= obj_X/budget and density_gain_Y  >= obj_Y/budget:

            if density_gain_X >= density_gain_Y:
                obj_X += gains_X[node]
                gain_adjustment(graph=graph,gains=gains_X,selected_element=node,spins=spins_X)
                
                X.append(node)
            else:
                obj_Y += gains_Y[node]
                gain_adjustment(graph=graph,gains=gains_Y,selected_element=node,spins=spins_Y)
                Y.append(node)

        elif density_gain_X  >= obj_X/budget:
            obj_X += gains_X[node]
            gain_adjustment(graph=graph,gains=gains_X,selected_element=node,spins=spins_X)
            X.append(node)
        
        elif density_gain_Y  >= obj_Y/budget:
            obj_Y += gains_Y[node]
            gain_adjustment(graph=graph,gains=gains_X,selected_element=node,spins=spins_Y)
            Y.append(node)

    sprint(obj_X)
    sprint(obj_Y)
    X_prime = []
    c_X = 0
    for idx in range(len(X)-1,-1,-1):

        if node_weights[X[idx]] + c_X <= budget:
            X_prime.append(X[idx])
            c_X += node_weights[X[idx]] 

        else:
            break

    sprint(c_X)


    Y_prime = []
    c_Y = 0
    for idx in range(len(Y)-1,-1,-1):

        if node_weights[Y[idx]] + c_Y <= budget:
            Y_prime.append(Y[idx])
            c_Y += node_weights[Y[idx]]
        else:
            break

    sprint(c_Y)

    obj_X_prime = calculate_obj(graph=graph,solution=X_prime)
    obj_Y_prime = calculate_obj(graph=graph,solution=Y_prime)

    if  obj_X_prime >= obj_Y_prime and obj_X_prime >= max_gain:

        return X_prime
    
    elif obj_X_prime <= obj_Y_prime and obj_Y_prime >= max_gain:

        return Y_prime
    
    else:
        return [max_singleton]
    


def DLA(graph,node_weights,budget,eps=0.1):


    gains = get_gains(graph=graph,ground_set=None)
    S_prime = LA(graph=graph,gains=gains,node_weights=node_weights,budget=budget)

    tau = calculate_obj(graph=graph,solution=S_prime)

    sprint(tau)

    eps_prime = eps / 14

    delta = np.ceil(np.log(1/eps_prime)/eps_prime)

    theta = 19 * tau / (6*eps_prime*budget)
    sprint(theta)

    # # X = set()
    # # Y = set ()

    X = []
    Y = []

    gains_X = gains.copy()
    gains_Y = gains.copy()

    # obj_X = 0 
    # obj_Y = 0

    c_X = 0 
    c_Y = 0

    spins_X = np.zeros(graph.number_of_nodes())
    spins_Y = np.zeros(graph.number_of_nodes())


    sprint(tau*(1-eps_prime)/(6*budget))

    while theta >= tau*(1-eps_prime)/(6*budget):
        sprint(theta)

        for node in graph.nodes():

            if node not in X or node not in Y:

                density_gain_X = gains_X[node]/node_weights[node]
                density_gain_Y = gains_Y[node]/node_weights[node]

                if node_weights[node]+ c_X <= budget and node_weights[node]+ c_Y <= budget:

                    if density_gain_X >= density_gain_Y:
                        gain_adjustment(graph=graph,gains=gains_X,selected_element=node,spins=spins_X)
                        X.append(node)
                        c_X += node_weights[node]
                        # X.add(node) 
                    else:
                        gain_adjustment(graph=graph,gains=gains_Y,selected_element=node,spins=spins_Y)
                        # Y.add(node) 
                        Y.append(node)
                        c_Y += node_weights[node]

                elif node_weights[node]+ c_X <= budget:
                    gain_adjustment(graph=graph,gains=gains_X,selected_element=node,spins=spins_X)
                    # X.add(node) 
                    X.append(node)
                    c_X += node_weights[node]
                
                elif node_weights[node]+ c_Y <= budget:
                    gain_adjustment(graph=graph,gains=gains_X,selected_element=node,spins=spins_Y)
                    # Y.add(node) 
                    # Y.append (node)
                    c_Y += node_weights[node]
        
        theta = (1-eps_prime) * theta


    # def calculate_l_prime(nodes, node_weights, bound):
    #     c = 0
    #     l_prime = []
        
    #     for node in nodes:
    #         if c + node_weights[node] <= bound:
    #             c += node_weights[node]
    #             l_prime.append(node)
    #         else:
    #             break
        
    #     return l_prime, c
    
    # def get_max_obj(graph, solution, node_weights, budget, c_value):
    #     max_obj = 0
    #     for node in graph.nodes():
    #         if node_weights[node] + c_value <= budget:
    #             max_obj = max(max_obj, calculate_obj(graph=graph, solution=solution + [node]))
    #     return max_obj


    # S = max(calculate_obj(graph=graph,solution=S_prime),calculate_obj(graph=graph,solution=X),calculate_obj(graph=graph,solution=Y))
    
    # sprint(S)
    # sprint(delta)

    # sprint(calculate_obj(graph=graph,solution=S_prime))
    # sprint(calculate_obj(graph=graph,solution=X))
    # sprint(calculate_obj(graph=graph,solution=Y))
    # for l in range(int(delta+1)):
    #     sprint(l)

    #     bound = eps_prime*budget * (1+eps_prime)** l

    #     X_l_prime,c_X = calculate_l_prime(X, node_weights, bound)
    #     Y_l_prime,c_Y = calculate_l_prime(Y, node_weights, bound)

    #     S = max(S,get_max_obj(graph=graph,solution=X_l_prime,node_weights=node_weights
    #                           ,budget=budget,c_value=c_X))
        
    #     S = max(S,get_max_obj(graph=graph,solution=Y_l_prime,node_weights=node_weights
    #                           ,budget=budget,c_value=c_Y))
    #     sprint(S)
        

    # return S


        

        



        



        # pass



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default='Facebook', help="Name of the dataset to be used (default: 'Facebook')")
    parser.add_argument("--budget", type=int,default=100, help="Budgets")
    parser.add_argument("--cost_model",type= str, default= 'degree', help = 'model of node weights')
    


    args = parser.parse_args()

    graph = nx.read_edgelist(f'../../data/snap_dataset/{args.dataset}.txt', 
                             create_using=nx.Graph(), nodetype=int)
    
    cost_model = args.cost_model
    node_weights = generate_node_weights(graph=graph,cost_model=cost_model)


    
    print(DLA(graph=graph,budget=args.budget,node_weights=node_weights))
    

