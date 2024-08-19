from utils import *
from helper_functions import *

def select_variable(gains):
    sum_gain = sum(gains.values())
    if sum_gain==0:
        return None
    else:
        prob_dist=[gains[key]/sum_gain for key in gains]
        element=np.random.choice([key for key in gains], p=prob_dist)
        return element
    

def get_gains(graph,ground_set):
    if ground_set is None:

        gains={node:graph.degree(node)+1 for node in graph.nodes()}
    else:
        print('A ground set has been given')
        gains={node:graph.degree(node)+1 for node in ground_set}
        print('Size of the ground set = ',len(gains))

    
    return gains

    
def gain_adjustment(graph,gains,selected_element,uncovered):
     

    if uncovered[selected_element]:
        gains[selected_element] -= 1
        uncovered[selected_element] = False
        for neighbor in graph.neighbors(selected_element):
            if neighbor in gains and gains[neighbor]>0:
                gains[neighbor]-=1

    for neighbor in graph.neighbors(selected_element):
        if uncovered[neighbor]:
            uncovered[neighbor]=False
            
            if neighbor in gains:
                gains[neighbor]-=1
            for neighbor_of_neighbor in graph.neighbors(neighbor):
                if neighbor_of_neighbor  in gains:
                    gains[neighbor_of_neighbor ]-=1


    assert gains[selected_element] == 0, f'gains of selected element = {gains[selected_element]}'


def prob_greedy(graph,budget,ground_set=None,delta=0):


    gains = get_gains(graph,ground_set)

    solution = []
    uncovered = defaultdict(lambda: True)


    for _ in range(budget):

        selected_element = select_variable(gains)

        if selected_element is None or gains[selected_element]<delta:
            break
        solution.append(selected_element)
        gain_adjustment(graph,gains,selected_element,uncovered)

    assert len(solution)<= budget, f'Number of elements in solution : {len(solution)}'
    return solution


def greedy(graph,budget,ground_set=None):
    
    number_of_queries = 0

    gains = get_gains(graph,ground_set)
    
    solution=[]
    uncovered=defaultdict(lambda: True)
    obj_val = 0

    for i in range(budget):
        number_of_queries += (len(gains)-i)

        selected_element=max(gains, key=gains.get)

        if gains[selected_element]==0:
            print('All elements are already covered')
            break
        solution.append(selected_element)

        obj_val += gains[selected_element]
        
        gain_adjustment(graph,gains,selected_element,uncovered)
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







