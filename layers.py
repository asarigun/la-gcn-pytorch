import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import numpy as np
import torch.nn as nn

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
        self.weight_0 = Parameter(torch.FloatTensor(in_features, out_features))

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.mask = []
        self.weights_mask0 = Parameter(torch.FloatTensor(2*in_features, in_features))
        print("Loading weight shape:")
        print(self.weights_mask0.shape)
        #self.weights_mask = nn.Module.get_parameter("weights_mask", shape=[2*in_features,in_features], dtype=torch.float32, initializer='uniform')

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight_0.size(1))
        self.weight_0.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    

    def forward(self, input, adj):

        input_new = []
        for i in range(len(self.add_all)):
            #listeyle x i torh.gather'da birleştirebilirmiyiz
            #t=np.random.randint(1,10,[4,5])
            #print("Loading input:", input)
            #print("Loading shape of input:",input.shape)
            #print("Loading type of [i]:", type([i]))
            #index = np.array([i])
            """"print("Loading index:",index)
                                                            index = torch.from_numpy(index)
                                                            print("Loading index:", index)
                                                            index0 = np.array([i]) 
                                                            print("Loading index:", index)"""
            """index = torch.from_numpy(index).long()
                                                print("Loading index:" , index)
                                                #index = index.unsqueeze(1)
                                                print("Loading index:" , index)
                                                index = index.expand(1,1,16) 
                                                print("Loading index:",index.shape)"""
            #index = index.squeeze()
            #aa_0= torch.tensor([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])
            index = torch.tensor([[i]*input.shape[1]])
            #print("Loading torch.tensor([[i]*input.shape[1]]):", torch.tensor([[i]*input.shape[1]]))
            aa = torch.gather(input, 0, torch.tensor([[i]*input.shape[1]])) #converting from tf to torch, what is index here?
            #print("Loading aa" , aa)
            #print("Loading shape of aa:" , aa.shape)
            aa_tile = torch.tile(aa, [len(self.add_all[i]), 1]) #expand central 
            #print("Loading aa_tile:" , aa_tile)
            #bb_nei_index = np.array(self.add_all[i])
            bb_nei_index2 = self.add_all[i]
            #print("Loading bb_nei_index2", bb_nei_index2)
            bb_nei_index2 = np.array([[i]*input.shape[1] for i in bb_nei_index2], dtype="int64")
            bb_nei_index2 = torch.tensor(bb_nei_index2)
            #print("Loading bb_nei_index2:", bb_nei_index2)
            #print("Loading bb_nei_index2", type(bb_nei_index2))
            #print("Loading bb_nei_index:" ,bb_nei_index)
            #bb_nei_index = torch.from_numpy(bb_nei_index)
            #print("Loading bb_nei_index:" , bb_nei_index)
            #bb_nei_index = np.array(self.add_all[i]) 
            #print("Loading bb_nei_index:", bb_nei_index)
            #bb_nei_index = torch.from_numpy(bb_nei_index).long()
            #print("Loading bb_nei_index:", bb_nei_index)
            #bb_nei_index = bb_nei_index.unsqueeze(1)
            #print("Loading bb_nei_index:", bb_nei_index)
            bb_nei = torch.gather(input,0, torch.tensor(bb_nei_index2)) #converting from tf to torch, what is index here?
            #print("Loading bb_nei_shape:", bb_nei.shape)
            cen_nei = torch.cat([aa_tile, bb_nei],1)
            #print("Loading cen_nei:", cen_nei)
                                      
            #mask0 = dot(cen_nei, self.weights_mask, sparse = self.sparse_inputs)
            mask0 = torch.mm(cen_nei, self.weights_mask0) #, sparse = self.sparse_inputs
            mask0 = self.Sig(mask0)
            #mask = nn.Dropout(mask0, 1-self.dropout)
                                      
            self.mask.append(mask0)
                                      
            new_cen_nei = aa + torch.sum(mask0 * bb_nei, 0, keepdims=True) #hadamard product of neighbors' feature vector and mask aggregator, then applying sum aggregator
            input_new.append(new_cen_nei)                                      
        #print("Loading input_new:", type(input_new[0]))
        #print(self.add_all)
        #input_new = torch.FloatTensor(input_new)  
        #input_new = np.array(input_new)
        #input_new = [t.numpy() for t in input_new]
        #input_new = torch.Tensor(input_new)
        input_new = torch.stack(input_new)                                     
        input_new = torch.squeeze(input_new)
        #print(input_new.shape)
        #print(self.weight.shape)
        #pre_sup = dot(input_new, self.weight) #sparse=self.sparse_inputs
        #return self.act(pre_sup)
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

"""import torch
import numpy as np
t=np.random.randint(1,10,[4,5])
t = torch.from_numpy(t)
print(",,,,,,,")
print(t)
index = np.array([1,2,2]) 
print("index:")
print(index)
index = torch.from_numpy(index).long()
print("-------")
print(index)
index = index.unsqueeze(1) 
print("........")
print(index)
index = index.expand(1,3,5) 
print("_____________")
print(index)
index = index.squeeze()
print("*****")
print(index)
#print(index,t,sep='\n')
t = torch.gather(t, 0, index)
#print("gathered:")
print("#########")
print(t)"""
