from argparse import ArgumentParser
from utils import *
import pandas as pd
from collections import defaultdict
import numpy as np
import os


#TO
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
        gains={node:graph.degree(node) for node in ground_set}

    return gains

    
def gain_adjustment(graph,gains,selected_element,spins):

    gains[selected_element]=-gains[selected_element]

    for neighbor in graph.neighbors(selected_element):

        if neighbor in gains:
            gains[neighbor]+=(2*spins[neighbor]-1)*(2-4*spins[selected_element])

    spins[selected_element]=1-spins[selected_element]

        

def prob_greedy(graph,budget,ground_set=None,delta=0):


    gains=get_gains(graph,ground_set)

    solution=[]
    # uncovered=defaultdict(lambda: True)
    # spins=np.ones(graph.number_of_nodes())
    spins={node:1 for node in graph.nodes()}


    for _ in range(budget):

        selected_element=select_variable(gains)

        if selected_element is None or gains[selected_element]<delta:
            break

        # print(gains[selected_element])

        solution.append(selected_element)
        gain_adjustment(graph,gains,selected_element,spins)

        # gains adjustment

        

    # print('Solution:',solution)
    # print('Degree:',[ graph.degree(node) for node in solution])
    # print('Coverage:',covered/graph.number_of_nodes())

    return solution



def greedy(graph,budget,ground_set=None):

    number_of_queries=0
    gains=get_gains(graph,ground_set)
    solution=[]
    spins={node:1 for node in graph.nodes()}

    for i in range(budget):
        number_of_queries += len(gains)
        selected_element=max(gains, key=gains.get)

        if gains[selected_element]==0:
            print('No more greedy actions are left')
            break

        # print(gains[selected_element])

        solution.append(selected_element)
        gain_adjustment(graph,gains,selected_element,spins)

    
    # print('Degree:',[ graph.degree(node) for node in solution])

    return solution,number_of_queries

        


if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument( "--dataset", type=str, default='Facebook', help="Name of the dataset to be used (default: 'Facebook')" )
    parser.add_argument( "--budgets", nargs='+', type=int, help="Budgets" )

    args = parser.parse_args()



    file_path=f'../../data/test/{args.dataset}'
    
    graph=load_from_pickle(file_path)

    
    df=defaultdict(list)

    for budget in args.budgets:

        solution=greedy(graph=graph,budget=budget)

        cut=calculate_cut(graph,solution)

        df['Dataset'].append(args.dataset)
        df['Dataset Path'].append(file_path)
        df['Budget'].append(budget)
        df['Solution'].append(solution)
        df['Objective Value'].append(cut)
        df['Objective Value (Ratio)'].append(cut/graph.number_of_edges())

    df=pd.DataFrame(df)

    print(df)

    save_folder='data/greedy'
    file_path=os.path.join(save_folder,args.dataset)
    os.makedirs(save_folder,exist_ok=True)
    save_to_pickle(df,file_path)

    








