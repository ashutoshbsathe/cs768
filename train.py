from argparse import ArgumentParser
from model import GraphClassification
from datamodule import MNISTSuperpixelsDataModule
import pytorch_lightning as pl

pl.seed_everything(1618, workers=True)

parser = ArgumentParser()
parser = GraphClassification.add_model_specific_args(parser)
args = parser.parse_args()
model = GraphClassification(**vars(args))
mnistsuperpixels = MNISTSuperpixelsDataModule(batch_size=64)

expt_name = f'conv={args.conv_operator}_hidden={args.hidden_channels}_out={args.out_channels}_L={args.num_layers}_H={args.num_attention_heads}_pool={args.pool_operator}'

trainer = pl.Trainer(
    gpus=1, 
    max_epochs=100, 
    track_grad_norm=2,
    callbacks=[
        pl.callbacks.ModelCheckpoint(
            dirpath='./models/' + expt_name,
            monitor='val_acc', mode='max',
        )
    ],
    logger=pl.loggers.TensorBoardLogger(
        save_dir='./models/' + expt_name,
    )
)
trainer.fit(model, mnistsuperpixels)
trainer.test(test_dataloaders=mnistsuperpixels.test_dataloader())
