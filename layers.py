import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(Module):


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

    def __init__(self, add_all, in_features, out_features, bias=False): #bias = True
        super(gcnmask, self).__init__()
        self.in_features = in_features
        self.Sig = nn.Sigmoid()
        self.out_features = out_features
        self.add_all = add_all
        self.drop_rate = 0.5
        self.weight_0 = Parameter(torch.FloatTensor(in_features, out_features))

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.mask = []
        self.weights_mask0 = Parameter(torch.FloatTensor(2*in_features, in_features))

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight_0.size(1))
        self.weight_0.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    
    def forward(self, input, adj):

        input_new = []
        for i in range(len(self.add_all)):

            index = torch.tensor([[i]*input.shape[1]])
            aa = torch.gather(input, 0, torch.tensor([[i]*input.shape[1]])) 
            aa_tile = torch.tile(aa, [len(self.add_all[i]), 1]) #expand central 
            bb_nei_index2 = self.add_all[i]
            bb_nei_index2 = np.array([[i]*input.shape[1] for i in bb_nei_index2], dtype="int64")
            bb_nei_index2 = torch.tensor(bb_nei_index2)
            bb_nei = torch.gather(input,0, torch.tensor(bb_nei_index2)) 
            cen_nei = torch.cat([aa_tile, bb_nei],1)
            mask0 = torch.mm(cen_nei, self.weights_mask0) 
            mask0 = self.Sig(mask0)
            mask0 = F.dropout(mask0, self.drop_rate)
                                      
            self.mask.append(mask0)
                                      
            new_cen_nei = aa + torch.sum(mask0 * bb_nei, 0, keepdims=True) #hadamard product of neighbors' features  and mask aggregator, then applying sum aggregator
            input_new.append(new_cen_nei)                                      
            
        input_new = torch.stack(input_new)                                     
        input_new = torch.squeeze(input_new)
        support = torch.mm(input_new, self.weight_0)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output                                                              
                                      
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
