import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch_geometric 
import torch_geometric.nn as nn_geo
import pytorch_lightning as pl

name_to_conv_operator = {
    'GAT': nn_geo.GAT,
    'GCN': nn_geo.GCN,
    'GraphSAGE': nn_geo.GraphSAGE,
}

name_to_pooling_operator = {
    'add': nn_geo.global_add_pool,
    'mean': nn_geo.global_mean_pool,
    'max': nn_geo.global_max_pool,
}

class GraphClassification(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.conv = name_to_conv_operator[kwargs['conv_operator']](
            in_channels=3, # TODO: parameterize with node2vec dim
            hidden_channels=kwargs['hidden_channels'],
            out_channels=kwargs['out_channels'],
            num_layers=kwargs['num_layers'],
            dropout=kwargs['dropout'],
            norm=nn.BatchNorm1d(kwargs['hidden_channels']),
        )
        self.pool = name_to_pooling_operator[kwargs['pool_operator']]
        output_dim = kwargs['hidden_channels'] if kwargs['out_channels'] is None else kwargs['out_channels']
        self.classifier = nn.Sequential(
            nn.Linear(output_dim, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 10)
        )
        self.save_hyperparameters()
        self.loss = nn.CrossEntropyLoss()
        print(self.hparams)

    def forward(self, batch):
        # batch is a `torch_geometric.data.Batch` operator
        # batch.x = nn_geo.Node2Vec(batch.edge_index, embedding_dim=8, walk_length=32, context_size=8).forward().to(self.device) # TODO: Can you get current device in PyTorch ?
        embeds = self.conv(batch.x, batch.edge_index)
        pooled = self.pool(embeds, batch.batch)
        logits = self.classifier(pooled)
        return logits 

    def training_step(self, batch, batch_idx):
        x = batch 
        y = batch.y
        logits = self.forward(x)
        loss = self.loss(logits, y)
        self.log('train_loss', loss)
        return loss 
    
    def eval_batch(self, batch, mode=None):
        x = batch
        y = batch.y
        logits = self.forward(x)
        loss = self.loss(logits, y)
        acc = (logits.argmax(dim=1) == y).sum().item() / logits.size(0)
        if mode is not None:
            self.log(f'{mode}_loss', loss)
            self.log(f'{mode}_acc', acc)

    def validation_step(self, batch, batch_idx):
        self.eval_batch(batch, 'val')

    def test_step(self, batch, batch_idx):
        self.eval_batch(batch, 'test')

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        lr_sched = torch.optim.lr_scheduler.StepLR(
            optim,
            step_size=20,
            gamma=0.2,
        )
        return [optim], [lr_sched]
