import numpy as np
import pickle as pkl
import networkx as nx
from scipy import sparse
import scipy.sparse as sp
import scipy.io as scio
from scipy.sparse import identity
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import pdb
import torch

##############################################
# Modified from https://github.com/tkipf/gcn #
##############################################

"""
ind.dataset.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
ind.dataset.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
ind.dataset.allx => the feature vectors of both labeled and unlabeled training instances 
    (a superset of ind.dataset.x) as scipy.sparse.csr.csr_matrix object;
    
ind.dataset.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
ind.dataset.ty => the one-hot labels of the test instances as numpy.ndarray object;
ind.dataset.ally => the labels for instances in ind.dataset.allx as numpy.ndarray object;

ind.dataset.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict object;
ind.dataset.test.index => the indices of test instances in graph, for the inductive setting as list object.

All objects above must be saved using python pickle module.
    
To cora example:
 ind.dataset.x => eigenvectors of training examples, a class object scipy.sparse.csr.csr_matrix, shape: (140, 1433)
 ind.dataset.tx => eigenvector test case, shape: (1000, 1433)
 ind.dataset.allx => + None None tagged training examples tag feature vector is ind.dataset.x superset, shape: (1708, 1433)

 ind.dataset.y => tag of training examples, hot encoded, numpy.ndarray class instance, the object is numpy.ndarray, shape: (140, 7)
 ind.dataset.ty => tag test examples, hot encoded, numpy.ndarray class instance, shape: (1000, 7)
 ind.dataset.ally => ind.dataset.allx corresponding label, one-hot encoding, shape: (1708, 7)

 ind.dataset.graph => map data, collections.defaultdict class instance, the format {index: [index_of_neighbor_nodes]}
 ind.dataset.test.index => test case id, 2157 OK

 These documents must be stored with the pickle module of python
"""


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset): 
    """Load data."""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']

    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)


    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)
    
    
    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = list(range(min(test_idx_reorder), max(test_idx_reorder)+1))
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()

    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
        
    if dataset == 'cora':
            
        idx_train = range(len(y)+1068)
        idx_val = range(len(y)+1068,len(y)+1068+500 )


    elif dataset == 'citeseer':   

        idx_train = range(len(y)+1707)
        idx_val = range(len(y)+1707, len(y)+1707+500)           
     
        
    elif dataset == 'pubmed':   
        idx_train = range(len(y)+18157)     
        idx_val = range(len(y)+18157, len(y)+18157+500)
   
    
    ## find each node's neighbors
    add_all = []
    for i in range(adj.shape[0]):
        add_all.append(adj[i].nonzero()[1])

    features = torch.FloatTensor(np.array(features.todense()))
    if dataset=="citeseer":
    	new_labels = []
    	for lbl in labels:
    		lbl = np.where(lbl==1)[0]
    		new_labels.append(lbl[0] if list(lbl)!=[] else 0)
    	labels = torch.LongTensor(new_labels)
    else:
    	labels = torch.LongTensor(np.where(labels)[1])
    
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)   

    return add_all, adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

