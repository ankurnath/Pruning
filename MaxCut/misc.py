
from argparse import ArgumentParser
import networkx as nx

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument( "--dataset", type=str, default='Facebook', help="Name of the dataset to be used (default: 'Facebook')" )
    parser.add_argument( "--budget", type=int, default=100, help='Budget' )
    parser.add_argument( "--size", type=int, default=750, help='size' )
    args = parser.parse_args()

    graph=nx.read_edgelist(f'../../data/snap_dataset/{args.dataset}.txt', create_using=nx.Graph(), nodetype=int)


    N = graph.number_of_nodes()

    greedy = 0

    x = 0
    for i in range(args.budget):
        greedy += (N-i)
        x += (args.size -i)

    print(round(x/greedy,4)*100)

