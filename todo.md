1. Train all the 12 models
2. On the best GAT model, fix the number of layers and vary number of attention heads 
3. On the best GAT model, fix number of layers and best number of attention heads and try varying the hidden layer size
4. Compare best GCN model's embeddings to best GraphSAGE model's embeddings to best GAT model's embeddings (do PCA)
5. On the best overall model, do neighbour sampling using the pooled embeddings to show retrieval performance
