import json
from model import GraphClassification 
from datamodule import MNISTSuperpixelsDataModule
import pytorch_lightning as pl 

grid_ckpts = [
    # GCN 
    './models/conv=GCN_hidden=256_out=None_L=2_H=1_pool=mean/epoch=92-step=72725.ckpt',
    './models/conv=GCN_hidden=256_out=None_L=4_H=1_pool=mean/epoch=70-step=55521.ckpt',
    './models/conv=GCN_hidden=256_out=None_L=8_H=1_pool=mean/epoch=90-step=71161.ckpt',
    './models/conv=GCN_hidden=256_out=None_L=16_H=1_pool=mean/epoch=84-step=66469.ckpt',
    # GraphSAGE 
    './models/conv=GraphSAGE_hidden=256_out=None_L=2_H=1_pool=mean/epoch=80-step=63341.ckpt',
    './models/conv=GraphSAGE_hidden=256_out=None_L=4_H=1_pool=mean/epoch=43-step=34407.ckpt',
    './models/conv=GraphSAGE_hidden=256_out=None_L=8_H=1_pool=mean/epoch=41-step=32843.ckpt',
    './models/conv=GraphSAGE_hidden=256_out=None_L=16_H=1_pool=mean/epoch=44-step=35189.ckpt',
    # GAT
    './models/conv=GAT_hidden=256_out=None_L=2_H=1_pool=mean/epoch=71-step=56303.ckpt',
    './models/conv=GAT_hidden=256_out=None_L=4_H=1_pool=mean/epoch=70-step=55521.ckpt',
    './models/conv=GAT_hidden=256_out=None_L=8_H=1_pool=mean/epoch=99-step=78199.ckpt',
    './models/conv=GAT_hidden=256_out=None_L=16_H=1_pool=mean/epoch=13-step=10947.ckpt',
]
dim_ckpts = [
    './models/conv=GCN_hidden=64_out=None_L=4_H=1_pool=mean/epoch=99-step=78199.ckpt',
    './models/conv=GCN_hidden=128_out=None_L=4_H=1_pool=mean/epoch=97-step=76635.ckpt',
    './models/conv=GCN_hidden=512_out=None_L=4_H=1_pool=mean/epoch=90-step=71161.ckpt',
    './models/conv=GCN_hidden=1024_out=None_L=4_H=1_pool=mean/epoch=75-step=59431.ckpt',
]
head_ckpts = [
    './models/conv=GAT_hidden=256_out=None_L=4_H=2_pool=mean/epoch=76-step=60213.ckpt',
    './models/conv=GAT_hidden=256_out=None_L=4_H=4_pool=mean/epoch=66-step=52393.ckpt',
    './models/conv=GAT_hidden=256_out=None_L=4_H=8_pool=mean/epoch=83-step=65687.ckpt',
    './models/conv=GAT_hidden=256_out=None_L=4_H=16_pool=mean/epoch=46-step=36753.ckpt',
]

results = {}

mnist = MNISTSuperpixelsDataModule(batch_size=64)
mnist.setup()

train = mnist.train_dataloader()
val = mnist.val_dataloader()
test = mnist.test_dataloader()

trainer = pl.Trainer(gpus=1)
for ckpt in grid_ckpts:
    model = GraphClassification.load_from_checkpoint(ckpt)
    model.eval()
    
    res_train = trainer.test(model=model, test_dataloaders=train)[0]
    res_val = trainer.test(model=model, test_dataloaders=val)[0]
    res_test = trainer.test(model=model, test_dataloaders=test)[0]

    key = '_'.join([model.hparams.conv_operator, str(model.hparams.hidden_channels), str(model.hparams.num_layers), str(model.hparams.num_attention_heads)])
    results[key] = {}
    results[key]['loss'] = (res_train['test_loss'], res_val['test_loss'], res_test['test_loss'])
    results[key]['accs'] = (res_train['test_acc'], res_val['test_acc'], res_test['test_acc'])
for ckpt in dim_ckpts:
    model = GraphClassification.load_from_checkpoint(ckpt)
    model.eval()
    
    res_train = trainer.test(model=model, test_dataloaders=train)[0]
    res_val = trainer.test(model=model, test_dataloaders=val)[0]
    res_test = trainer.test(model=model, test_dataloaders=test)[0]

    key = '_'.join([model.hparams.conv_operator, str(model.hparams.hidden_channels), str(model.hparams.num_layers), str(model.hparams.num_attention_heads)])
    results[key] = {}
    results[key]['loss'] = (res_train['test_loss'], res_val['test_loss'], res_test['test_loss'])
    results[key]['accs'] = (res_train['test_acc'], res_val['test_acc'], res_test['test_acc'])
for ckpt in head_ckpts:
    model = GraphClassification.load_from_checkpoint(ckpt)
    model.eval()
    
    res_train = trainer.test(model=model, test_dataloaders=train)[0]
    res_val = trainer.test(model=model, test_dataloaders=val)[0]
    res_test = trainer.test(model=model, test_dataloaders=test)[0]

    key = '_'.join([model.hparams.conv_operator, str(model.hparams.hidden_channels), str(model.hparams.num_layers), str(model.hparams.num_attention_heads)])
    results[key] = {}
    results[key]['loss'] = (res_train['test_loss'], res_val['test_loss'], res_test['test_loss'])
    results[key]['accs'] = (res_train['test_acc'], res_val['test_acc'], res_test['test_acc'])

with open('./results.json', 'w') as f:
    json.dump(results, f, indent=4, sort_keys=True)
