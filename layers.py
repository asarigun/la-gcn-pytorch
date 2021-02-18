import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

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
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class gcnmask(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(gcnmask, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        #self.add_all = add_all
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.mask = []

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
        input_new = []
        for i in range(len(self.add_all)):
            aa = torch.gather(input, [i])
            aa_tile = torch.tile(aa, [len(self.add_all[i]), 1]) #expand central
            bb_nei = torch.gather(input,self.add_all[i])
            cen_nei = torch.cat([aa_tile, bb_nei],1)
                                      
            mask0 = dot(cen_nei, self.W, sparse = self.sparse_inputs)
            mask0 = nn.Sigmoid(mask0)
            mask = nn.Dropout(mask0, 1-self.dropout)
                                      
            self.mask.append(mask)
                                      
            new_cen_nei = aa + torch.sum(mask * bb_nei, 0, keepdims=True) #hadamard product of neighbors' feature vector and mask aggregator, then applying sum aggregator
            x_new.append(new_cen_nei)
                                      
        input_new = torch.squeeze(input_new)
        pre_sup = dot(input_new, self.W, sparse=self.sparse_inputs)
                                      
        return self.act(pre_sup)                               
                                      
                                      
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
