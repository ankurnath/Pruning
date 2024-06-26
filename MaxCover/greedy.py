from argparse import ArgumentParser
from utils import *
import pandas as pd
from collections import defaultdict
import numpy as np
import os

#TO
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
        gains={node:graph.degree(node)+1 for node in ground_set}

    return gains

    
def gain_adjustment(graph,gains,selected_element,uncovered):

    # print('Gains:',gains[selected_element])

    # uncovered[selected_element]=False
    # for neighbor in graph.neighbors(selected_element):
    #     uncovered[neighbor]=False

    # for node in gains:
    #     gains[node]= 1 if uncovered[node] else 0
    #     for neighbor in graph.neighbors(node):
    #         if uncovered[neighbor]:
    #             gains[node]+=1
            

    if uncovered[selected_element]:
        gains[selected_element]-=1
        uncovered[selected_element]=False
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


    assert gains[selected_element]==0


def prob_greedy(graph,budget,ground_set=None,delta=0):


    gains=get_gains(graph,ground_set)

    solution=[]
    uncovered=defaultdict(lambda: True)


    for _ in range(budget):

        selected_element=select_variable(gains)

        if selected_element is None or gains[selected_element]<delta:
            break
        solution.append(selected_element)
        gain_adjustment(graph,gains,selected_element,uncovered)
    return solution



def greedy(graph,budget,ground_set=None):


    gains=get_gains(graph,ground_set)


    solution=[]
    uncovered=defaultdict(lambda: True)

    for _ in range(budget):

        selected_element=max(gains, key=gains.get)

        if gains[selected_element]==0:
            print('All elements are already covered')
            break
        solution.append(selected_element)
        gain_adjustment(graph,gains,selected_element,uncovered)


    return solution

        


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        default='Facebook',
        help="Name of the dataset to be used (default: 'Facebook')"
    )

    parser.add_argument(
        "--budgets",
        nargs='+',
        type=int,
        help="Budgets"
    )
  
    args = parser.parse_args()

    file_path=f'../../data/test/{args.dataset}'
    
    graph=load_from_pickle(file_path)

    
    df=defaultdict(list)

    for budget in args.budgets:

        solution=greedy(graph=graph,budget=budget)

        cover=calculate_cover(graph,solution)

        df['Budget'].append(budget)
        df['Objective Value'].append(cover)
        df['Objective Value (Ratio)'].append(cover/graph.number_of_nodes())
        df['Solution'].append(solution)
        
        
    df=pd.DataFrame(df)
    print(df)
    save_folder=f'data/{args.dataset}'
    file_path=os.path.join(save_folder,'Greedy')
    os.makedirs(save_folder,exist_ok=True)
    save_to_pickle(df,file_path)








