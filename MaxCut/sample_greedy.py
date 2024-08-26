from utils import *
from helper_functions import *



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

@njit
def get_gains(adj_list,start,end):

    N = len(start)
    gains = np.zeros(N)

    for node in range(N):
        gains[node]=end[node]-start[node]

    return gains


@njit
def gain_adjustment(adj_list,start,end,gains,selected_element,spins):

    gains[selected_element] = - gains[selected_element]

    for neighbor in adj_list[start[selected_element]:end[selected_element]]:

        gains[selected_element] += (2*spins[neighbor]-1)*(2-4*spins[selected_element])

    spins[selected_element] = 1 - spins[selected_element]


@njit
def select_element(gains,node_weights,mask):

    N = len(gains)

    max_gain_ratio = 0
    selected_element = -1
    for node in range(N):
        if mask[node] == 1 and gains[node]/node_weights[node] >= max_gain_ratio:
            max_gain_ratio = gains[node]/node_weights[node]
            selected_element = node

    mask [selected_element] = 0
    return selected_element





def sample_greedy(graph:nx.Graph,budget:int,node_weights:dict,ground_set=None):
    
    # forward mapping
    graph, forward_mapping, reverse_mapping = relabel_graph(graph=graph)
    node_weights= np.array([node_weights[reverse_mapping[node]] for node in reverse_mapping])

    adj_list,start,end = flatten_graph(graph=graph)

    start_time = time.time()

    # solution = get_solution(adj_list,start,end,node_weights,budget)
    # solution = np.nonzero(array)[0]
    N = graph.number_of_nodes()
    
    gains = get_gains(adj_list=adj_list,start=start,end=end)
    spins = np.ones(N)
    
    if ground_set is None:
        ground_set_size = N

        mask = np.ones(N)
    else:
        ground_set_size = len(ground_set)
        mask = np.zeros(N)
        for element in ground_set:
            mask[forward_mapping[element]] = 1

    
    
    # max singleton
            
    
    # Initialize variables
    max_gain = 0
    max_node = None

    # Iterate through each node

    number_of_queries = 0
    for node in range(ground_set_size):
        # Check if the node is in the mask, within budget, and has a higher gain than the current max
        if mask[node] == 1 and node_weights[node] <= budget:
            number_of_queries += 1
            if  gains[node] > max_gain:
                max_gain = gains[node]
                max_node = node
            

    solution = []

    start_time = time.time()

    constraint = 0

    p = np.sqrt(2)-1 
    # p = 0.5
    for i in tqdm(range(ground_set_size)):
        number_of_queries += (ground_set_size- i)
        selected_element = select_element(gains,node_weights,mask)
        

        if selected_element != -1 and gains[selected_element] !=0 and node_weights[selected_element] + constraint <= budget:
            
            if random.random() < p: 
            # if random.random() > p :
            
                constraint += node_weights[selected_element]
                gain_adjustment(adj_list=adj_list,start=start,end=end,
                            gains=gains,selected_element=selected_element,spins=spins)
                
                solution.append(selected_element)

    end_time = time.time()

    print("Elapsed time:",round(end_time-start_time,4))
    
    if calculate_obj(graph,solution)>=calculate_obj(graph,[max_node]):
        return calculate_obj(graph,solution),number_of_queries,[reverse_mapping[node] for node in solution]
    else:
        return calculate_obj(graph,[max_node]),number_of_queries,[reverse_mapping[max_node]]


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default='Facebook', help="Name of the dataset to be used (default: 'Facebook')")
    parser.add_argument("--budget", type=int,default=100, help="Budgets")
    parser.add_argument("--cost_model",type= str, default= 'degree', help = 'model of node weights')
    


    args = parser.parse_args()
    

    # graph = nx.read_edgelist(f'../../data/snap_dataset/{args.dataset}.txt', create_using=nx.Graph(), nodetype=int)
    # cost_model = args.cost_model
    # node_weights = generate_node_weights(graph=graph,cost_model=cost_model)
    # budget = args.budget


    graph = nx.erdos_renyi_graph(n=5000, p =0.02)
    node_weights = {node:random.random() for node in graph.nodes()}
    # node_weights = {node:1 for node in graph.nodes()}
    # sprint(sum(node_weights.values()))
    # budget = 0.2 *sum(node_weights.values.sum()) 
    budget = 0.02 * sum(node_weights.values())

    # print(sample_greedy(graph=graph,budget=budget,node_weights=node_weights)[0])
    print(sample_greedy(graph=graph,budget=budget,node_weights=node_weights)[0])

    # sample_greedy(graph=graph,budget=args.budget,node_weights=node_weights)
    


