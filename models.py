import torch.nn as nn
import torch.nn.functional as F
from layers import *


class GCN_MASK(nn.Module):
    def __init__(self, add_all, nfeat, nhid, nclass, dropout, device):
        super(GCN_MASK, self).__init__()
        
        self.weight0 = nn.Parameter(torch.empty(nfeat, nhid, device = device)).to(device)
        self.bias0 = nn.Parameter(torch.empty(nhid, device = device)).to(device)

        self.weight1 = nn.Parameter(torch.empty(nhid, nclass, device =device)).to(device)
        self.bias1 = nn.Parameter(torch.empty(nclass, device=device)).to(device)

        self.weights_mask0 =nn.Parameter(torch.empty(2*nhid, nhid, device=device)).to(device)
        self.parameters = nn.ParameterList([self.weight0, self.bias0, self.weight1, self.bias1, self.weights_mask0])

        self.add_all = add_all

        self.gc1 = GraphConvolution(nfeat, nhid, self.weight0, self.bias0, device) 
        self.gc2 = gcnmask(add_all, nhid, nclass, self.weight1, self.bias1, self.weights_mask0, device)

        self.dropout = dropout
        
    def _mask(self):
        return self.mask        

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)
