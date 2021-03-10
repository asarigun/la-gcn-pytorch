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
import tensorflow as tf

# flags = tf.app.flags
# FLAGS = flags.FLAGS

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


def load_data(fastgcn_setting, dataset_str, train_percente, attack_dimension,train_jump):
    """Load data."""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    #pdb.set_trace()
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)


    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
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


    if fastgcn_setting == 1:   
          
        #pdb.set_trace()   
        
        if dataset_str == 'cora':
            
            idx_train = range(len(y)+1068)
            idx_val = range(len(y)+1068,len(y)+1068+500 )
            print("==== the fastgcn dataset split for cora ====", len(idx_train))

        elif dataset_str == 'citeseer':   

            idx_train = range(len(y)+1707)
            idx_val = range(len(y)+1707, len(y)+1707+500)           
            print("==== the fastgcn dataset split for citeseer ====", len(idx_train)) 
        
        elif dataset_str == 'pubmed':   
            idx_train = range(len(y)+18157)     
            idx_val = range(len(y)+18157, len(y)+18157+500)
            print("==== the fastgcn dataset split for pubmed ====", len(idx_train)) 
    else:


        print('===choose the training data as propotio===', train_percente)
        train_number = int(train_percente * labels.shape[0])
        idx_train = range(train_number)
        idx_val = range(train_number, train_number+500)

    train_mask = sample_mask(idx_train, labels.shape[0]) # the training index is true, others is false
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
    
    ## find each node's neighbors
    add_all = []
    for i in range(adj.shape[0]):
        add_all.append(adj[i].nonzero()[1])
        #print(i)
        #print(add_all)
    #print("Loading add_all:")
    #print(add_all)
    if attack_dimension > 0:     
        print('====the attacked dimention====', attack_dimension)  

    # attack node featues (the random  dimension)
        at_d = attack_dimension 
        for i in range(features.shape[0]):

            at_idx = np.random.choice(features.shape[1], size=at_d, replace=False)
            idex_fea = features[i, at_idx].toarray()
            at_fea = np.where ( idex_fea==0,1,0 )
            features[i,at_idx] = at_fea

    return add_all, adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features,adj):
    """Row-normalize feature matrix and convert to tuple representation"""
    #rowsum = np.ones(shape=features.sum(1).shape)
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features) ## type(features) scipy.sparse.csr.csr_matrix'

    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)

    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
##  add the node self feature and normalize the primary adjcency matrix


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""


    #adj_normalized0 = normalize_adj(adj + sp.eye(adj.shape[0]))
    adj_normalized0 = normalize_adj(adj+sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized0)


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})

    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))
    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)
