import numpy as np
import pandas as pd
from PIL import ImageColor
import networkx as nx
from matplotlib import pyplot as plt

from torchvision.datasets import MNIST
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

mnist_dataset = MNIST('./MNIST', train=True)
graph_dataset = MNISTSuperpixels(root='./MNISTSuperpixels', train=True)

idx = 10240
image, label = mnist_dataset[idx]
graph = graph_dataset[idx]
print(f'Image label={label}')
visualize(image, graph)
