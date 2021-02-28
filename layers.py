import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


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

    def __init__(self, add_all, in_features, out_features, bias=True):
        super(gcnmask, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.add_all = add_all
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.mask = []
        self.weights_mask = nn.Module.get_parameter("weights_mask", shape=[2*in_features,in_features], dtype=torch.float32, initializer='uniform')

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):

        input_new = []
        for i in range(len(self.add_all)):
            aa = torch.gather(input, [i])
            aa_tile = torch.tile(aa, [len(self.add_all[i]), 1]) #expand central
            bb_nei = torch.gather(input,self.add_all[i])
            cen_nei = torch.cat([aa_tile, bb_nei],1)
                                      
            #mask0 = dot(cen_nei, self.weights_mask, sparse = self.sparse_inputs)
            mask0 = torch.mm(cen_nei, self.weights_mask) #, sparse = self.sparse_inputs
            mask0 = nn.Sigmoid(mask0)
            mask = nn.Dropout(mask0, 1-self.dropout)
                                      
            self.mask.append(mask)
                                      
            new_cen_nei = aa + torch.sum(mask * bb_nei, 0, keepdims=True) #hadamard product of neighbors' feature vector and mask aggregator, then applying sum aggregator
            x_new.append(new_cen_nei)
                                      
        input_new = torch.squeeze(input_new)
        #pre_sup = dot(input_new, self.weight) #sparse=self.sparse_inputs
        #return self.act(pre_sup)
        support = torch.mm(input_new, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output                                                              
                                      
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
"""
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
"""
        #self.weights_mask = nn.Module.get_parameter("weights_mask", shape=[2*in_features,in_features], dtype=torch.float32, initializer='uniform')
"""
        with tf.variable_scope(self.name + '_vars'):
            
            self.vars['weights_0'] = glorot([input_dim, output_dim], name = 'weights_0')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

            self.vars['weights_mask0'] = tf.get_variable(name = 'weights_mask0', shape=[2*input_dim,input_dim],\dtype=tf.float32, initializer = tf.contrib.layers.xavier_initializer()) 
"""
