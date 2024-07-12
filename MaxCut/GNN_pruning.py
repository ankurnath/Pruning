import os
import torch

import numpy as np
from utils import *

from torch_geometric.utils.convert import  from_networkx

dataset = 'Skitter'

graph=load_from_pickle(f'../../data/train/{dataset}')



data= from_networkx(graph)


from greedy import greedy

solution,_=greedy(graph=graph,budget=100,ground_set=None)

# subgraph = make_subgraph(graph,solution)


# train_mask=torch.tensor([mapping[node] for node in subgraph.nodes()],dtype=torch.long)
mapping = dict(zip(graph.nodes(), range(graph.number_of_nodes())))
train_mask = torch.tensor([mapping[node] for node in solution], dtype=torch.long)
y=torch.zeros(graph.number_of_nodes(),dtype=torch.long)

# for node in subgraph.nodes():
for node in solution:
    # train_mask[mapping[node]]=1
    y[mapping[node]]=1


# data.train_mask=train_mask

# print('y:',torch.sum(y))
data.y=y
num_features=1

x=[graph.degree(node) for node in graph.nodes()]
# data.x= torch.randn(size=(graph.num,num_features))
# data.x = torch.tensor(x,dtype=torch.float).reshape(-1,1)
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
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

model = GCN(hidden_channels=16)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

model.train()

for epoch in range(1,1000):

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

test_graph = load_from_pickle(f'../../data/test/{dataset}')
test_data = from_networkx(test_graph)

test_data.x =torch.rand(size=(test_graph.number_of_nodes(),1))

out = model(test_data.x, test_data.edge_index)
# print(out[:100])
pred = out.argmax(dim=1).cpu().numpy()  # Use the class with highest probability.
indices = np.where(pred == 1)[0]

print('Pruned Ground ratio:',indices.shape[0]/test_graph.number_of_nodes())

reverse_mapping = dict(zip(range(test_graph.number_of_nodes()),test_graph.nodes()))

ground_set=[ reverse_mapping[node] for node in indices]

solution_greedy,_ =greedy(test_graph,budget=100)

solution_pruned,_= greedy(test_graph,budget=100,ground_set=ground_set)


print('Ratio:',calculate_cut(test_graph, solution_pruned)/calculate_cut(test_graph, solution_greedy))


print(solution_pruned,solution_greedy)

# # print(pred.sum())
# correct = pred == data.y


# positive = pred==1

# negative = pred== 0
# # print()

# # print('Correct positive:',(correct & positive).sum())
# # Print the count of correct positive predictions
# print('Correct positive:', (correct & positive).sum().item())

# print('Positive class',positive.sum().item())
# print('Negative class',negative.sum().item())
# print(correct.sum()/graph.number_of_nodes())
# print(subgraph_size.sum()/data.y.sum())

# print(subgraph_size.sum()/graph.number_of_nodes())




