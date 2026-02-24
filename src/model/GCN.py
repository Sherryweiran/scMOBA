import math
import numpy as np
import torch
import torch.nn as nn 
import torch.nn.init as init
import torch.nn.functional as F 
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


def aff_to_adj(x, y=None):
    x = x.detach().cpu().numpy()
    adj = np.matmul(x, x.transpose())
    adj +=  -1.0*np.eye(adj.shape[0])
    adj_diag = np.sum(adj, axis=0) #rowise sum
    adj = np.matmul(adj, np.diag(1/adj_diag))
    adj = adj + np.eye(adj.shape[0])
    adj = torch.Tensor(adj).cuda()

    return adj

import torch

def aff_to_adj_batch(x):
    # x has shape (batch_size, num_nodes, feature_dim)
    x = x.detach()
    x = nn.functional.normalize(x, dim=-1)

    batch_size, num_nodes, feature_dim = x.shape
    
    # 计算批量中所有图的相似度矩阵：x @ x^T
    similarity = torch.bmm(x, x.transpose(1, 2))  # Shape: (batch_size, num_nodes, num_nodes)
    
    # 减去单位矩阵，以避免自连接
    identity_matrix = torch.eye(num_nodes, device=x.device).unsqueeze(0).expand(batch_size, -1, -1)

    adj = similarity - identity_matrix  # Shape: (batch_size, num_nodes, num_nodes)
    
    # 计算每列的和（即每个节点与其他节点的连接强度总和）

    adj_diag = adj.sum(dim=1, keepdim=True)  # Shape: (batch_size, 1, num_nodes)
    
    # 进行列归一化：adj = adj * (1 / adj_diag)
    # 由于 adj_diag 是 (batch_size, 1, num_nodes)，需要广播到 (batch_size, num_nodes, num_nodes)
    adj = adj / (adj_diag + 1e-8)  # 小值epsilon以防除零
    
    # 添加单位矩阵，确保每个节点的自连接是 1
    adj = adj + identity_matrix  # Shape: (batch_size, num_nodes, num_nodes)
    adj = torch.Tensor(adj).cuda()
    
    return adj


"""class GraphConvolution(Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.bmm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'"""
class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 初始化为二维张量 (in_features, out_features)
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        
        if bias:
            # 初始化偏置为 (out_features,)
            self.bias = Parameter(torch.FloatTensor(out_features))  
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()

    def reset_parameters(self):
        # 初始化权重和偏置
        stdv = 1. / math.sqrt(self.in_features)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # input: (batch_size, num_nodes, in_features)
        # adj: (batch_size, num_nodes, num_nodes)
        batch_size, num_nodes, in_features = input.size()

        # 动态扩展 weight 和 bias 的形状
        weight = self.weight.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, in_features, out_features)
        
        if self.bias is not None:
            bias = self.bias.unsqueeze(0).expand(batch_size, -1)  # (batch_size, out_features)
        else:
            bias = None

        # 1. 执行批量矩阵乘法: input (batch_size, num_nodes, in_features) 和 weight (batch_size, in_features, out_features)
        support = torch.bmm(input, weight)  # (batch_size, num_nodes, out_features)

        # 2. 执行邻接矩阵乘法: adj (batch_size, num_nodes, num_nodes) 和 support (batch_size, num_nodes, out_features)
        output = torch.bmm(adj, support)  # (batch_size, num_nodes, out_features)

        # 3. 如果存在偏置，将其添加到输出中
        if bias is not None:
            output = output + bias.unsqueeze(1)  # (batch_size, num_nodes, out_features)

        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        #self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nout)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        feat = F.dropout(x, self.dropout, training=self.training)
        x = self.gc3(feat, adj)        
        #x = self.linear(x)
        # x = F.softmax(x, dim=1)
        return x