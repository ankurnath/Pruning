import os
import torch

import numpy as np
from utils import *
from argparse import ArgumentParser
from torch_geometric.utils.convert import  from_networkx
from greedy import greedy
from collections import defaultdict

import pandas as pd

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument( "--dataset", type=str, default='Facebook', help="Name of the dataset to be used (default: 'Facebook')" )
    parser.add_argument( "--budget", type= int , default= 100, help="Budget" )
    args = parser.parse_args()


    dataset = args.dataset
    budget = args.budget
    graph=load_from_pickle(f'../../data/train/{dataset}')
    data= from_networkx(graph)


    _,_,solution=greedy(graph=graph,budget=budget,ground_set=None)


    mapping = dict(zip(graph.nodes(), range(graph.number_of_nodes())))
    train_mask = torch.tensor([mapping[node] for node in solution], dtype=torch.long)
    y=torch.zeros(graph.number_of_nodes(),dtype=torch.long)


    for node in solution:
        y[mapping[node]]=1



    data.y=y
    num_features=1

    x=[graph.degree(node) for node in graph.nodes()]
    data.x = torch.rand(size=(graph.number_of_nodes(),1))


    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv

    class GCN(torch.nn.Module):
        def __init__(self, hidden_channels):
            super(GCN, self).__init__()
            torch.manual_seed(12345)
            self.conv1 = GCNConv(num_features, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, 2)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = x.relu()
            x = self.conv2(x, edge_index)
            return x

    model = GCN(hidden_channels=16)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()

    for epoch in tqdm(range(1,1000)):

        out = model(data.x, data.edge_index)  # Perform a single forward pass.

        # mask=torch.cat([train_mask,torch.randint(graph.number_of_nodes())],axis=0)
        mask = torch.cat([train_mask, torch.randint(0, train_mask.size(0), (train_mask.size(0),))], dim=0)
        # print('Mask size',train_mask.shape)
        # print('Mask size',mask.shape)

        # print(torch.sum(data.y[mask]))
        loss = criterion(out[mask], data.y[mask])  # Compute the loss solely based on the training nodes.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
        


    model.eval()


    
    test_graph = load_graph(f'../../data/snap_dataset/{dataset}.txt')
    test_data = from_networkx(test_graph)

    test_data.x =torch.rand(size=(test_graph.number_of_nodes(),1))

    start = time.time()
    out = model(test_data.x, test_data.edge_index)
    pred = out.argmax(dim=1).cpu().numpy()  # Use the class with highest probability.
    indices = np.where(pred == 1)[0]
    reverse_mapping = dict(zip(range(test_graph.number_of_nodes()),test_graph.nodes()))
    pruned_universe = [reverse_mapping[node] for node in indices]
    # print(pruned_universe)
    end= time.time()

    time_to_prune = end-start

    print('time elapsed to pruned',time_to_prune)

    graph = test_graph

    ##################################################################

    Pg=len(pruned_universe)/graph.number_of_nodes()
    start = time.time()
    objective_unpruned,queries_unpruned,solution_unpruned= greedy(graph,budget)
    end = time.time()
    time_unpruned = round(end-start,4)
    print('Elapsed time (unpruned):',round(time_unpruned,4))

    start = time.time()
    objective_pruned,queries_pruned,solution_pruned = greedy(graph=graph,budget=budget,ground_set=pruned_universe)
    end = time.time()
    time_pruned = round(end-start,4)
    print('Elapsed time (pruned):',time_pruned)
    
    
    ratio = objective_pruned/objective_unpruned


    print('Performance of GNNpruner')
    print('Size Constraint,k:',budget)
    print('Size of Ground Set,|U|:',graph.number_of_nodes())
    print('Size of Pruned Ground Set, |Upruned|:', len(pruned_universe))
    print('Pg(%):', round(Pg,4)*100)
    print('Ratio:',round(ratio,4)*100)
    print('Queries:',round(queries_pruned/queries_unpruned,4)*100)


    save_folder = f'data/{dataset}'
    os.makedirs(save_folder,exist_ok=True)
    save_file_path = os.path.join(save_folder,'GNNpruner')

    performance_ratios = []
    for budget in [10, 30, 50, 70, 90, 100]:
        
        unpruned_objective, unpruned_queries, unpruned_solution = greedy(graph, budget=budget)
        pruned_objective, pruned_queries, pruned_solution = greedy(graph=graph, budget=budget, ground_set=pruned_universe)
        
        performance_ratios.append(round(pruned_objective / unpruned_objective,4)) 

    df ={      'Dataset':dataset,'Budget':budget,'Objective Value(Unpruned)':objective_unpruned,
              'Objective Value(Pruned)':objective_pruned ,'Ground Set': graph.number_of_nodes(),
              'Ground set(Pruned)':len(pruned_universe), 'Queries(Unpruned)': queries_unpruned,'Time(Unpruned)':time_unpruned,
              'Time(Pruned)': time_pruned,
              'Queries(Pruned)': queries_pruned, 'Pruned Ground set(%)': round(Pg,4)*100,
              'Ratio(%)':round(ratio,4)*100, 'Queries(%)': round(queries_pruned/queries_unpruned,4)*100,
              'TimeRatio': time_pruned/time_unpruned,
              'TimeToPrune':time_to_prune,
              'Multibudget':[performance_ratios]


              }

   
    df = pd.DataFrame(df,index=[0])
    save_to_pickle(df,save_file_path)
    # print(df)
    print(df['Multibudget'])






