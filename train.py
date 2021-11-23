from model import GraphClassification
from datamodule import MNISTSuperpixelsDataModule
import pytorch_lightning as pl

model = GraphClassification(
    conv_operator='GAT',
    hidden_channels=64,
    out_channels=None,
    num_layers=4,
    dropout=0,
    pool_operator='mean',
    lr=5e-3,
    weight_decay=1e-5
)
mnistsuperpixels = MNISTSuperpixelsDataModule(batch_size=64)

trainer = pl.Trainer(gpus=1, max_epochs=100, track_grad_norm=2)
trainer.fit(model, mnistsuperpixels)
trainer.test(test_dataloaders=mnistsuperpixels.test_dataloader())
