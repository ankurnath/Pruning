from argparse import ArgumentParser
from utils import load_from_pickle,calculate_cover
import pandas as pd
from collections import defaultdict




def greedy(graph,budget):

    

    gains={node:graph.degree(node) for node in graph.nodes()}

    solution=[]
    uncovered=defaultdict(lambda: True)
    covered=0

    for _ in range(budget):

        selected_element=max(gains, key=gains.get)

        if gains[selected_element]==0:
            print('All elements are already covered')
            break

        # print(gains[selected_element])

        solution.append(selected_element)

        # gains adjustment

        if uncovered[selected_element]:
            covered+=1
            gains[selected_element]-=1
            uncovered[selected_element]=False
            for neighbour in graph.neighbors(selected_element):
                if gains[neighbour]>0:
                    gains[neighbour]-=1

        for neighbour in graph.neighbors(selected_element):
            if uncovered[neighbour]:
                uncovered[neighbour]=False
                
                if  neighbour in gains:
                    gains[neighbour]-=1
                
                covered+=1
                for neighbour_of_neighbour in graph.neighbors(neighbour):

                    gains[neighbour_of_neighbour]-=1

    # print('Solution:',solution)
    # print('Degree:',[ graph.degree(node) for node in solution])
    # print('Coverage:',covered/graph.number_of_nodes())

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
        "--budget",
        type=int,
        default=20,
        help="Budget"
    )

    args = parser.parse_args()

    graph=load_from_pickle(f'../../data/test/{args.dataset}')

    solution=greedy(graph=graph,budget=args.budget)

    print('Greedy Coverage:',calculate_cover(graph,solution)/graph.number_of_nodes())








