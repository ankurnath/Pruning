from utils import *


def flatten_graph(graph:nx.Graph):
    flat_adj_matrix = []
    
    n = graph.number_of_nodes()
    start = [0 for _ in range(n)]
    end = [0 for _ in range(n)]
    adj_list_dict = nx.to_dict_of_lists(graph)

    

    for node, neighbors in adj_list_dict.items():
        start[node] = len(flat_adj_matrix)
        end[node] = start[node] + len(neighbors)
        flat_adj_matrix += neighbors
    return np.array(flat_adj_matrix), np.array(start), np.array(end)

# @njit 
def get_gains(start,end):

    N = len(start)
    gains = np.zeros(N)

    for node in range(N):
        gains[node]=end[node]-start[node]+1

    return gains


# @njit 
def gain_adjustment(graph,gains,selected_element,uncovered):
    adj_list,start,end = graph
    if uncovered[selected_element] ==1 :
        gains[selected_element]-=1
        uncovered[selected_element]= 0
        for neighbor in adj_list[start[selected_element]:end[selected_element]]:
            gains[neighbor]-=1

    for neighbor in adj_list[start[selected_element]:end[selected_element]]:
        if uncovered[neighbor] == 1:
            uncovered[neighbor]= 0
            gains[neighbor]-=1
            for neighbor_of_neighbor in adj_list[start[neighbor]:end[neighbor]]:
                gains[neighbor_of_neighbor ]-=1


    assert gains[selected_element] == 0

# @njit 
def estimate_opt(graph,node_weights,ground_set):
    print('Estimate Optimal')

    adj_list,start,end = graph

    gains = get_gains(start=start,end=end)

    N = len(start)

    S = np.zeros(N)
    uncovered = np.ones(N)

    size_S = 0

    obj_val = 0
    for node in range(N):
        if ground_set[node] == 1 and gains[node]/node_weights[node] >= obj_val:
            S[size_S] = 0
            size_S +=1
            obj_val += gains[node]
            gain_adjustment(graph=graph,gains=gains,selected_element=node,uncovered=uncovered)

    sprint(obj_val)
    sprint(size_S)
    return obj_val/4


# @njit 
def fast_threshold_greedy(graph,node_weights,ground_set,eps,alpha):

    gamma = estimate_opt(graph=graph,node_weights=node_weights,ground_set=ground_set)

    tau = 8*alpha*gamma
    h =0
    
    adj_list ,start,end =graph

    N = len(start)

    S = np.zeros(shape=(N,),dtype=np.int32)

    size_S = 0

    S_prime = np.zeros(shape=(N,),dtype=np.int32)

    c_S =0

    e = 2.718281

    gains = get_gains(start,end)

    while tau > (1-eps)*gamma/e:

        for node in range(N):
            if ground_set[node] == 1 and S_prime[node] == 0 and node_weights[node]+c_S <=1 and gains[node]/node_weights[node]>= tau:
                S[size_S] = node
                S_prime[node] = 1
                size_S += 1
                c_S += node_weights[node]

        tau  = (1-eps) * tau

    sprint(size_S)
    sprint(S[:size_S])
    print([end[node]-start[node] for node in S[:size_S]])
    return S,size_S,S_prime





def calculate_obj(graph,solution):

    adj_list,start,end = graph

    N = len(start)

    covered = np.zeros(N)

    for node in range(N):
        if solution[node] == 1:
            covered[node] = 1
            for neighbor in adj_list[start[node]:end[node]]:
                covered[neighbor] = 1

    return covered.sum()


# @njit 
def fast_threshold_greedy_post_processing(graph,node_weights,ground_set,eps):

    adj_list,start,end = graph
    N = len(start)

    S,size_S,S_prime = fast_threshold_greedy(graph=graph,node_weights=node_weights,
                                    ground_set=ground_set,eps=eps,alpha=1/eps)
    
    # sprint(S_prime.sum())
    sprint(S_prime.nonzero())
    
    max_obj_val = calculate_obj(graph=graph,solution= S_prime)
    sprint(max_obj_val)

    gains = get_gains(start,end)

    for node in range(N):
        # Check if the node is in the mask, within budget, and has a higher gain than the current max
        if ground_set[node] == 1 and node_weights[node] <= 1:
            if  gains[node] > max_obj_val:
                max_obj_val = gains[node]

    
    upper_bound = int(np.floor(np.log(1/eps) /np.log(1+eps)))
    sprint(upper_bound)
    for i in range(upper_bound+1):

        cost_bound = eps *(1+eps)**i
        # sprint(cost_bound)
        c_S_k =0
        S_k = np.zeros(shape=(N,),dtype=np.int32)
        covered = np.zeros(N)
        for j in range(size_S):
            if c_S_k + node_weights[S[j]] <= cost_bound:
                c_S_k += node_weights[S[j]]
                S_k[S[j]] = 1
                covered[S[j]] = 1
                for neighbor in adj_list[start[S[j]]:end[S[j]]]:
                    covered[neighbor] = 1

            else:
                break

        obj_val = calculate_obj(graph=graph,solution=S_k)
        
        for node in range(N):
            if ground_set[j] ==1 and c_S_k+node_weights[node]<=1:
                gain = 1- covered[node]
                for neighbor in adj_list[start[node]:end[node]]:
                    gain += 1- covered[neighbor]


                max_obj_val = max(max_obj_val,obj_val+gain)

    return max_obj_val


        
def feldman(graph:nx.Graph,node_weights:dict,budget,eps =0.01,ground_set=None):

    graph, forward_mapping, reverse_mapping = relabel_graph(graph=graph)
    N = graph.number_of_nodes()

    graph = flatten_graph(graph=graph)

    node_weights= np.array([node_weights[reverse_mapping[node]]/budget for node in reverse_mapping])

    if ground_set is None:
        mask = np.ones(N)
    else:
        mask = np.zeros(N)
        for element in ground_set:
            mask[forward_mapping[element]] = 1


    for i in range(N):
        if node_weights[i]>1:
            mask[i] = 0


    return fast_threshold_greedy_post_processing(graph=graph,node_weights=node_weights
                                                 ,ground_set=mask,eps=eps)

    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default='Facebook', help="Name of the dataset to be used (default: 'Facebook')")
    parser.add_argument("--budget", type=int,default=10, help="Budgets")
    parser.add_argument("--cost_model",type= str, default= 'uniform', help = 'model of node weights')
    


    args = parser.parse_args()

    graph = nx.read_edgelist(f'../../data/snap_dataset/{args.dataset}.txt', create_using=nx.Graph(), nodetype=int)
    
    cost_model = args.cost_model
    node_weights = generate_node_weights(graph=graph,cost_model=cost_model)

    print(feldman(graph=graph,budget=args.budget,node_weights=node_weights))
        



        


        





    



    pass





