import argparse
import random

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, ChebConv, TransformerConv
import torch.optim as optim
from AKGNN_model import AKGNN
import torch.nn.functional as F

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
setup_seed(42)
# 命令行参数
def parse_arguments():
    parser = argparse.ArgumentParser(description='Base-Graph Neural Network')
    parser.add_argument('--dataset', choices=['Cora', 'Citeseer', 'Pubmed'], default='Cora',
                        help="Dataset selection")
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden layer dimension')
    parser.add_argument('--dropout_rate', type=float, default=0.6, help='Dropout rate')
    parser.add_argument('--model', choices=['GCN', 'GAT', 'SAGE', 'ChebNet', 'TransformerConv'], default='GCN',
                        help="Model selection")
    parser.add_argument('--lr', default=0.05, help="Learning Rate selection")
    parser.add_argument('--wd', default=5e-4, help="weight_decay selection")
    parser.add_argument('--epochs', default=10000, help="train epochs selection")
    parser.add_argument('--layers', default=6, help="layer number")
    parser.add_argument('--tsne_drawing', choices=[True, False], default=False,
                        help="Whether to use tsne drawing")
    parser.add_argument('--tsne_colors', default=['#ffc0cb', '#bada55', '#008080', '#420420', '#7fe5f0', '#065535', '#ffd700'], help="colors")
    return parser.parse_args()


# 加载数据集
def load_dataset(name):
    dataset = Planetoid(root='dataset/' + name, name=name, transform=T.NormalizeFeatures())
    return dataset

# 使用Tsne绘图
def plot_points(z, y):
    z = TSNE(n_components=2).fit_transform(z.detach().cpu().numpy())
    classes = len(torch.unique(y))
    y = y.cpu().numpy()
    plt.figure(figsize=(8, 8))
    for i in range(classes):
        plt.scatter(z[y == i, 0], z[y == i, 1], s=20, color=args.tsne_colors[i])
    plt.axis('off')
    plt.savefig('{} embeddings ues tnse to plt figure.png'.format(args.model))
    plt.show()



def train(model, data):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.adj)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def feature_shuffling(data):
    num_nodes = data.x.size(0)
    perm = torch.randperm(num_nodes)
    shuffled_x = data.x[perm]
    return shuffled_x

def test(model, data):
    model.eval()
    logits, accs = model(data.x, data.adj), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs, logits
def adj_coo(data):
    shape = torch.Size([data.x.shape[0], data.x.shape[0]])
    indices = data.edge_index
    edge_weight = torch.ones(data.edge_index.size(1), dtype=torch.float32).cuda()
    return torch.sparse.FloatTensor(indices, edge_weight, shape).cuda()
if __name__ == "__main__":
    args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_dataset(args.dataset)
    data = dataset[0].to(device)
    data.adj = adj_coo(data)
    model = AKGNN(n_layer=args.layers, in_dim=data.x.shape[1], h_dim=args.hidden_dim,
                  n_class=len(torch.unique(data.y)), activation=F.leaky_relu, dropout=args.dropout_rate).to(device)
    print(model)
    print(f"Loaded {args.dataset} dataset with {data.num_nodes} nodes and {data.num_edges} edges.")
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    Best_Acc = []
    for epoch in range(1, args.epochs):
        loss = train(model, data)
        accs, log= test(model, data)
        train_acc, val_acc, test_acc = accs
        print(f'Epoch: [{epoch:03d}/{args.epochs}], Loss: {loss:.4f}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')
        Best_Acc.append(test_acc)
    if args.tsne_drawing == True:
        plot_points(log, data.y)
    print('---------------------------')
    print('Best Acc: {:.4f}'.format(max(Best_Acc)))
    print('---------------------------')