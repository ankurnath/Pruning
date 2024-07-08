import os
import torch


from utils import *

from torch_geometric.utils.convert import  from_networkx

graph=load_from_pickle(f'../../data/train/DBLP')

mapping = dict(zip(graph.nodes(), range(graph.number_of_nodes())))

data= from_networkx(graph)


from greedy import greedy

solution=greedy(graph=graph,budget=100,ground_set=None)

subgraph = make_subgraph(graph,solution)


train_mask=torch.tensor([mapping[node] for node in subgraph.nodes()],dtype=torch.long)
y=torch.zeros(graph.number_of_nodes(),dtype=torch.long)

for node in subgraph.nodes():
    # train_mask[mapping[node]]=1
    y[mapping[node]]=1


# data.train_mask=train_mask
data.y=y
num_features=1

x=[graph.degree(node) for node in graph.nodes()]
# data.x= torch.randn(size=(graph.num,num_features))
data.x = torch.tensor(x,dtype=torch.float).reshape(-1,1)


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
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

model = GCN(hidden_channels=16)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

model.train()

for epoch in range(1, 3):

    out = model(data.x, data.edge_index)  # Perform a single forward pass.

    # mask=torch.cat([train_mask,torch.randint(graph.number_of_nodes())],axis=0)
    mask = torch.cat([train_mask, torch.randint(0, train_mask.size(0), (train_mask.size(0),))], dim=0)
    print('Mask size',train_mask.shape)
    print('Mask size',mask.shape)
    loss = criterion(out[mask], data.y[mask])  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    optimizer.zero_grad()  # Clear gradients.
    


model.eval()
out = model(data.x, data.edge_index)
pred = out.argmax(dim=1)  # Use the class with highest probability.
correct = pred == data.y


subgraph_size= pred==1


print(correct.sum()/graph.number_of_nodes())
print(subgraph_size.sum()/data.y.sum())




