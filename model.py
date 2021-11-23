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
        conv_kwargs = {
            'in_channels': 3, # Superpixels have 3D features
            'hidden_channels': kwargs['hidden_channels'],
            'out_channels': kwargs['out_channels'],
            'num_layers': kwargs['num_layers'],
            'dropout': kwargs['dropout'],
            'norm': nn.BatchNorm1d(kwargs['hidden_channels'])
        }
        if kwargs['conv_operator'] == 'GAT':
            conv_kwargs['heads'] = kwargs['num_attention_heads']
        self.conv = name_to_conv_operator[kwargs['conv_operator']](**conv_kwargs)
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
        self.loss = nn.CrossEntropyLoss()
        print(self.hparams)

    def forward(self, batch):
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
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('GraphClassificationModel')
        parser.add_argument('--conv_operator', type=str, choices=['GCN', 'GraphSAGE', 'GAT'], default='GAT', help='The graph convolution operator to use')
        parser.add_argument('--hidden_channels', type=int, default=256, help='The dimension of hidden node embeddings')
        parser.add_argument('--out_channels', type=int, default=None, help='The dimension of final embeddings. Pass `None` to make it equal to `hidden_channels`')
        parser.add_argument('--num_layers', type=int, default=8, help='Number of conv layers')
        parser.add_argument('--dropout', type=float, default=0, help='Dropout for conv layers')
        parser.add_argument('--num_attention_heads', type=int, default=1, help='Number of attention heads for GAT. Ignored if conv operator != GAT')
        parser.add_argument('--pool_operator', type=str, default='mean', choices=['mean', 'max', 'sum'], help='The global pooling operator to use')
        parser.add_argument('--lr', type=float, default=5e-3, help='Learning rate of AdamW optimizer')
        parser.add_argument('--weight_decay', type=float, default=1e-5, help='Learning rate of AdamW optimizer')
        return parent_parser
