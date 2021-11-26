import matplotlib.pyplot as plt 
import json 

with open('./results.json') as f:
    data = json.load(f)

for layer in ['GCN', 'GraphSAGE', 'GAT']:
    train_loss = []
    val_loss = []
    test_loss = []

    train_acc = []
    val_acc = []
    test_acc = []
    for num_layer in [2, 4, 8, 16]:
        key = f'{layer}_256_{num_layer}_1'
        train_loss.append(data[key]['loss'][0])
        val_loss.append(data[key]['loss'][1])
        test_loss.append(data[key]['loss'][2])

        train_acc.append(data[key]['accs'][0] * 100)
        val_acc.append(data[key]['accs'][1] * 100)
        test_acc.append(data[key]['accs'][2] * 100)
    
    plt.clf()
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 4))
    ax1.plot([2, 4, 8, 16], train_loss, label='Train set', marker='o')
    ax1.plot([2, 4, 8, 16], val_loss, label='Dev set', marker='^')
    ax1.plot([2, 4, 8, 16], test_loss, label='Test set', marker='*')
    ax1.set_xlabel('Number of layers')
    ax1.set_ylabel('Loss')
    ax1.set_xticks([2, 4, 8, 16])
    ax1.legend()

    ax2.plot([2, 4, 8, 16], train_acc, label='Train set', marker='o')
    ax2.plot([2, 4, 8, 16], val_acc, label='Dev set', marker='^')
    ax2.plot([2, 4, 8, 16], test_acc, label='Test set', marker='*')
    ax2.set_xlabel('Number of layers')
    ax2.set_ylabel('Accuracy')
    ax2.set_xticks([2, 4, 8, 16])
    ax2.legend()

    plt.savefig(f'plots/{layer}_loss_acc.pdf', bbox_inches='tight', dpi=300)

train_loss = []
val_loss = []
test_loss = []

train_acc = []
val_acc = []
test_acc = []
for num_heads in [1, 2, 4, 8, 16]:
    key = f'GAT_256_4_{num_heads}'
    train_loss.append(data[key]['loss'][0])
    val_loss.append(data[key]['loss'][1])
    test_loss.append(data[key]['loss'][2])

    train_acc.append(data[key]['accs'][0] * 100)
    val_acc.append(data[key]['accs'][1] * 100)
    test_acc.append(data[key]['accs'][2] * 100)

plt.clf()
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 4))
ax1.plot([1, 2, 4, 8, 16], train_loss, label='Train set', marker='o')
ax1.plot([1, 2, 4, 8, 16], val_loss, label='Dev set', marker='^')
ax1.plot([1, 2, 4, 8, 16], test_loss, label='Test set', marker='*')
ax1.set_xlabel('Number of heads')
ax1.set_ylabel('Loss')
ax1.set_xticks([1, 2, 4, 8, 16])
ax1.legend()

ax2.plot([1, 2, 4, 8, 16], train_acc, label='Train set', marker='o')
ax2.plot([1, 2, 4, 8, 16], val_acc, label='Dev set', marker='^')
ax2.plot([1, 2, 4, 8, 16], test_acc, label='Test set', marker='*')
ax2.set_xlabel('Number of heads')
ax2.set_ylabel('Accuracy')
ax2.set_xticks([1, 2, 4, 8, 16])
ax2.legend()

plt.savefig(f'plots/num_heads_loss_acc.pdf', bbox_inches='tight', dpi=300)

train_loss = []
val_loss = []
test_loss = []

train_acc = []
val_acc = []
test_acc = []
for dim in [64, 128, 256, 512, 1024]:
    key = f'GCN_{dim}_4_1'
    train_loss.append(data[key]['loss'][0])
    val_loss.append(data[key]['loss'][1])
    test_loss.append(data[key]['loss'][2])

    train_acc.append(data[key]['accs'][0] * 100)
    val_acc.append(data[key]['accs'][1] * 100)
    test_acc.append(data[key]['accs'][2] * 100)

plt.clf()
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 4))
ax1.plot([64, 128, 256, 512, 1024], train_loss, label='Train set', marker='o')
ax1.plot([64, 128, 256, 512, 1024], val_loss, label='Dev set', marker='^')
ax1.plot([64, 128, 256, 512, 1024], test_loss, label='Test set', marker='*')
ax1.set_xlabel('Dimension of embeddings')
ax1.set_ylabel('Loss')
ax1.set_xticks([64, 128, 256, 512, 1024])
ax1.set_xticklabels([str(x) for x in [64, 128, 256, 512, 1024]], rotation=45)
ax1.legend()

ax2.plot([64, 128, 256, 512, 1024], train_acc, label='Train set', marker='o')
ax2.plot([64, 128, 256, 512, 1024], val_acc, label='Dev set', marker='^')
ax2.plot([64, 128, 256, 512, 1024], test_acc, label='Test set', marker='*')
ax2.set_xlabel('Dimension of embeddings')
ax2.set_ylabel('Accuracy')
ax2.set_xticks([64, 128, 256, 512, 1024])
ax2.set_xticklabels([str(x) for x in [64, 128, 256, 512, 1024]], rotation=45)
ax2.legend()

plt.savefig(f'plots/dimensionality_loss_acc.pdf', bbox_inches='tight', dpi=300)
