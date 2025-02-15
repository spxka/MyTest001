import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.utils import to_dense_adj


class AKConv(torch.nn.Module):
    def __init__(self):
        super(AKConv, self).__init__()
        self.lambda_ = torch.nn.Parameter(torch.tensor(1.), requires_grad=True)
        self.special_spmm = SpecialSpmm()
        self.i = None
        self.v_1 = None
        self.v_2 = None
        self.nodes = None

    def get_actual_lambda(self):
        return 1 + torch.relu(self.lambda_)

    def forward(self, input, adj):
        lambda_ = self.get_actual_lambda()

        if self.i == None:
            self.nodes = adj.shape[0]
            dummy = [i for i in range(self.nodes)]
            i_1 = torch.tensor([dummy, dummy]).cuda()
            i_2 = adj.coalesce().indices().cuda()
            self.i = torch.cat([i_1, i_2], dim=1)
            self.v_1 = torch.tensor([1 for _ in range(self.nodes)]).cuda()
            self.v_2 = torch.tensor([1 for _ in range(len(i_2[0]))]).cuda()

        v_1 = ((2 * lambda_ - 2) / lambda_) * self.v_1
        v_2 = (2 / lambda_) * self.v_2
        v = torch.cat([v_1, v_2])
        e_rowsum = self.special_spmm(self.i, v, torch.Size([self.nodes, self.nodes]),
                                     torch.ones(size=(self.nodes, 1)).cuda())
        return self.special_spmm(self.i, v, torch.Size([self.nodes, self.nodes]), input).div(e_rowsum)


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""

    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)
class AKGNN(torch.nn.Module):
    def __init__(self, n_layer, in_dim, h_dim, n_class, activation, dropout):
        super(AKGNN, self).__init__()
        self.layers = torch.nn.ModuleList([AKConv() for _ in range(n_layer)])
        self.theta = torch.nn.Linear(in_dim, h_dim)
        self.predictor = torch.nn.Linear(h_dim * n_layer, n_class)
        self.activation = activation
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.theta.weight.size(1))
        self.theta.weight.data.uniform_(-stdv, stdv)
        self.theta.bias.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.predictor.weight.size(1))
        self.predictor.weight.data.uniform_(-stdv, stdv)
        self.predictor.bias.data.uniform_(-stdv, stdv)
    def u_coo(self, adj, shape):
        indices = adj
        edge_weight = torch.ones(adj.size(1), dtype=torch.float32).cuda()
        return torch.sparse.FloatTensor(indices, edge_weight, shape).cuda()
    def forward(self, input, adj):
        h = self.activation(self.theta(input))
        h = F.dropout(h, self.dropout, training=self.training)
        h_list = []
        for propagation_layer in self.layers:
            h = propagation_layer(h, adj)
            h = F.dropout(h, self.dropout, training=self.training)
            h_list.append(h)
        h = torch.cat(h_list, dim = 1)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.predictor(h)
        return F.log_softmax(h, dim=1)