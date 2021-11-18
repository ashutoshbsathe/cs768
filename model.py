import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch_geometric 
import torch_geometric.nn as nn_geo
import pytorch_lightning as pl

name_to_conv_operator = {
    'GAT': nn.GAT,
    'GCN': nn.GCN,
    'GraphSAGE': nn.GraphSAGE,
}

name_to_pooling_operator = {
    'add': nn_geo.global_add_pooling,
    'mean': nn_geo.global_mean_pooling,
    'max': nn_geo.global_max_pooling,
}

class GraphClassification(pl.LightningModule):
    def __init__(**kwargs):
        self.conv = name_to_conv_operator[conv_operator](
            in_channels=1, # dataset has only 1D features
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.pool = name_to_pooling_operator[pool_operator]
        output_dim = hidden_channels if out_channels is None else out_channels
        self.linear = nn.Linear(output_dim, 10)
        self.save_hyperparameters()

    def forward(self, batch):
        # batch is a `torch_geometric.data.Batch` operator 
        embeds = self.conv(batch.x, batch.edge_index)
        pooled = self.pool(embeds, batch.batch)
        logits = self.linear(pooled)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch 
        logits = self.forward(x)
        loss = F.nll_loss(logits, y)
        self.log('train_loss', loss)
        return loss 
    
    def eval(self, batch, mode=None):
        x, y = batch
        logits = self.forward(x)
        loss = F.nll_loss(logits, y)
        acc = (logits.argmax(dim=1) == y).sum().item() / logits.size(0)
        if mode is not None:
            self.log(f'{mode}_loss', loss)
            self.log(f'{mode}_acc', acc)

    def validation_step(self, batch, batch_idx):
        self.eval(batch, 'val')

    def test_step(self, batch, batch_idx):
        self.eval(batch, 'test')

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=0.9,
            weight_decay=self.hparams.weight_decay
        )
        lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim,
            mode='min',
            factor=0.1,
            patience=15,
            threshold=0.0001,
            min_lr=1e-8
        )
        return optim, lr_sched
