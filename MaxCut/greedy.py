from utils import *
from helper_functions import *

def select_variable(gains):
    positive_gains = {k: v for k, v in gains.items() if v > 0}
    
    # If no positive gains, return None
    if not positive_gains:
        return None
    
    # Calculate the sum of positive gains
    sum_gain = sum(positive_gains.values())
    
    # Calculate the probability distribution
    prob_dist = [v / sum_gain for v in positive_gains.values()]
    
    # Randomly select an element based on the probability distribution
    element = np.random.choice(list(positive_gains.keys()), p=prob_dist)

    return element
    

def get_gains(graph,ground_set):
    if ground_set is None:

        gains={node:graph.degree(node) for node in graph.nodes()}
    else:
        print('A ground set has been given')
        gains={node:graph.degree(node) for node in ground_set}
        print('Size of the ground set = ',len(gains))

    
    return gains

    
def gain_adjustment(graph,gains,selected_element,spins):

    gains[selected_element]=-gains[selected_element]

    for neighbor in graph.neighbors(selected_element):

        if neighbor in gains:
            gains[neighbor]+=(2*spins[neighbor]-1)*(2-4*spins[selected_element])

    spins[selected_element]=1-spins[selected_element]
     

    


def prob_greedy(graph,budget,ground_set=None,delta=0):


    gains = get_gains(graph,ground_set)

    solution = []
    # uncovered = defaultdict(lambda: True)

    spins={node:1 for node in graph.nodes()}


    for _ in range(budget):

        selected_element = select_variable(gains)

        if selected_element is None or gains[selected_element]<delta:
            break
        solution.append(selected_element)
        gain_adjustment(graph,gains,selected_element,spins)

    assert len(solution)<= budget, f'Number of elements in solution : {len(solution)}'
    return solution


def greedy(graph,budget,ground_set=None):
    
    number_of_queries = 0

    gains = get_gains(graph,ground_set)
    
    solution=[]
    # uncovered=defaultdict(lambda: True)
    spins={node:1 for node in graph.nodes()}
    obj_val = 0

    for i in range(budget):
        number_of_queries += (len(gains)-i)

        selected_element=max(gains, key=gains.get)

        if gains[selected_element]==0:
            print('All elements are already covered')
            break
        solution.append(selected_element)

        obj_val += gains[selected_element]
        
        gain_adjustment(graph,gains,selected_element,spins)
    print('Objective value =', obj_val)
    print('Number of queries =',number_of_queries)

    return obj_val,number_of_queries,solution






if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument( "--dataset", type=str, default='Facebook', help="Name of the dataset to be used (default: 'Facebook')" )
    parser.add_argument("--budget", type=int,default=5, help="Budgets")
  
    args = parser.parse_args()

    dataset = args.dataset
    budget = args.budget

    load_graph_file_path=f'../../data/snap_dataset/{dataset}.txt'

    graph = load_graph(load_graph_file_path)
    
    greedy(graph=graph,budget=budget)







