from model import GraphClassification
from datamodule import MNISTSuperpixelsDataModule
import pytorch_lightning as pl

model = GraphClassification(
    conv_operator='GCN',
    hidden_channels=128,
    out_channels=None,
    num_layers=1,
    dropout=0,
    pool_operator='mean',
    lr=1e-2,
    weight_decay=0
)
mnistsuperpixels = MNISTSuperpixelsDataModule()

trainer = pl.Trainer()
trainer.fit(model, mnistsuperpixels)