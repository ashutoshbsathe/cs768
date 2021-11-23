import numpy as np
import pandas as pd
from PIL import ImageColor
import networkx as nx
from matplotlib import pyplot as plt

import torch 
import torch.nn as nn 
from torchvision.datasets import MNIST
from torch_geometric.nn import GCN, GraphSAGE, GAT, global_max_pool
from torch_geometric.data import Batch
from torch_geometric.datasets import MNISTSuperpixels 

def visualize(image, graph):
    height = 8
    plt.figure(figsize=(2*height+1, height))

    plt.subplot(1, 2, 1)
    plt.title('MNIST')
    plt.imshow(np.array(image))

    plt.subplot(1, 2, 2)
    x, edge_index = graph.x, graph.edge_index
    
    # src: https://raw.githubusercontent.com/pyg-team/pytorch_geometric/83db552aba6f8cc53672189d424daa6917a25510/examples/mnist_visualization.py
    df = pd.DataFrame({'from': edge_index[0], 'to': edge_index[1]})
    G = nx.from_pandas_edgelist(df, 'from', 'to')

    # flip over the axis of pos, this is because the default axis direction of networkx is different
    pos = {i: np.array([graph.pos[i][0], 27 - graph.pos[i][1]]) for i in range(graph.num_nodes)}

    # get the current node index of G
    idx = list(G.nodes())

    # set the node sizes using node features
    size = x[idx] * 500 + 200

    # set the node colors using node features
    color = []
    for i in idx:
        grey = x[i]
        color.append('skyblue' if grey == 0 else 'red')

    nx.draw(G, with_labels=True, node_size=size, node_color=color, pos=pos)
    plt.title('MNIST Superpixel')

    plt.savefig('./visualization.pdf', dpi=300, bbox_inches='tight')

#mnist_dataset = MNIST('./MNIST', train=True)
graph_dataset = MNISTSuperpixels(root='./MNISTSuperpixels', train=True)

all_adjacency_list_lens = np.array([data.edge_index.size(1) for data in graph_dataset])
print(all_adjacency_list_lens.mean(), all_adjacency_list_lens.std())
all_node_feats = torch.stack([data.x for data in graph_dataset]).squeeze()
print(all_node_feats.mean(axis=0), all_node_feats.std(axis=0))
print(all_node_feats.std(axis=0).nonzero())
print(all_node_feats.mean(axis=0).nonzero())
exit(0)
idx = 10240
#image, label = mnist_dataset[idx]
graph = graph_dataset[idx]

gcn = GCN(in_channels=1, hidden_channels=128, num_layers=128)

print(graph)
batch = Batch.from_data_list([graph_dataset[0], graph_dataset[1], graph_dataset[2], graph_dataset[3]])
y = gcn(batch.x, batch.edge_index)
print(y.size())
y = global_max_pool(y, batch.batch)
print(y.size())
#print(f'Image label={label}')
#visualize(image, graph)
