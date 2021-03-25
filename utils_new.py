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


def load_data(dataset="cora"): #fastgcn_setting,, train_percente, attack_dimension,train_jump
    """Load data."""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    #pdb.set_trace()
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
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()

    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    #print(adj)
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
        
    if dataset == 'cora':
            
        idx_train = range(len(y)+1068)
        idx_val = range(len(y)+1068,len(y)+1068+500 )
        print("==== the fastgcn dataset split for cora ====", len(idx_train))

    elif dataset == 'citeseer':   

        idx_train = range(len(y)+1707)
        idx_val = range(len(y)+1707, len(y)+1707+500)           
        print("==== the fastgcn dataset split for citeseer ====", len(idx_train)) 
        
    elif dataset == 'pubmed':   
        idx_train = range(len(y)+18157)     
        idx_val = range(len(y)+18157, len(y)+18157+500)
        print("==== the fastgcn dataset split for pubmed ====", len(idx_train)) 
    """
    else:


        print('===choose the training data as propotio===', train_percente)
        train_number = int(train_percente * labels.shape[0])
        idx_train = range(train_number)
        idx_val = range(train_number, train_number+500)
"""

    """
    train_mask = sample_mask(idx_train, labels.shape[0]) # the training index is true, others is false
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
    """
    ## find each node's neighbors
    add_all = []
    for i in range(adj.shape[0]):
        add_all.append(adj[i].nonzero()[1])
        #print(i)
        #print(add_all)
    print("Loading add_all:")
    print(add_all)
    features = torch.FloatTensor(np.array(features.todense()))
    #print("Loading features:", features)
    labels = torch.LongTensor(np.where(labels)[1])
    #print("Loading labels:", labels)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    #print("Loading adj:", adj)
    idx_train = torch.LongTensor(idx_train)
    #print("Loading idx_train:", idx_train)
    idx_val = torch.LongTensor(idx_val)
    #print("Loading idx_val:", idx_val)
    idx_test = torch.LongTensor(idx_test)   
    #print("Loading idx_test:", idx_test) 
    """
    if attack_dimension > 0:     
        print('====the attacked dimention====', attack_dimension)  

    # attack node featues (the random  dimension)
        at_d = attack_dimension 
        for i in range(features.shape[0]):

            at_idx = np.random.choice(features.shape[1], size=at_d, replace=False)
            idex_fea = features[i, at_idx].toarray()
            at_fea = np.where ( idex_fea==0,1,0 )
            features[i,at_idx] = at_fea
    """
    return add_all, adj, features, labels, idx_train, idx_val, idx_test

    #add_all, adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask

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
