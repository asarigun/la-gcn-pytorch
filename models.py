import torch.nn as nn
import torch.nn.functional as F
from layers import *


class GCN_MASK(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN_MASK, self).__init__()
        #self.add_all = add_all

        self.gc1 = GraphConvolution(nfeat, nhid) #add_all = self.add_all
        self.gc2 = gcnmask(nhid, nclass)
        self.dropout = dropout
        
    def _mask(self):
        return self.mask        

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)
