import numpy as np
import scipy.sparse as sp
import torch
import networkx as nx

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(path="data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))

    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])
    
    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    """print("Loading idx:")
                print(idx)"""
    #print("Loading shape of idx:")
    #print(idx.shape)
    idx_map = {j: i for i, j in enumerate(idx)}
    #print("Loading shape of idx_map:")
    #print(len(idx_map))
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    #print("Loading shape of edges_unordered:")
    #print(edges_unordered.shape)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    #print("Loading shape of edges:")
    #print(edges.shape)
    #adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))          
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    #print(adj)
    adj = adj.tocsr() #sorun burdan kaynaklanıyor!!!
    #adj = adj.tolil()
    #print("Loading shape of adj:")
    #print(adj.shape)
    """print("Loading adj:")
                print(adj)"""
    add_all = []
    for i in range(adj.shape[0]):
        add_all.append(adj[i].nonzero()[1])
        #print("loading adj[i]:",adj[i])
        #print("loading adj[i].nonzero()[1]:", adj[i].nonzero()[1])
    #print("Loading shape of add_all:")
    #print(len(add_all))
    #print(add_all)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    #print("Loading features:", type(features))
    adj = normalize(adj + sp.eye(adj.shape[0]))
    """
    print('===choose the training data as propotio===', train_percente)
    train_number = int(train_percente * labels.shape[0])
    idx_train = range(train_number)
    idx_val = range(train_number, train_number+500)
    idx_test = 
    
    """
    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)
    
    
    features = torch.FloatTensor(np.array(features.todense()))
    print("Loading features:", features)
    labels = torch.LongTensor(np.where(labels)[1])
    print("Loading labels:", labels)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    print("Loading adj:", adj)
    idx_train = torch.LongTensor(idx_train)
    print("Loading idx_train:", idx_train)
    idx_val = torch.LongTensor(idx_val)
    print("Loading idx_val:", idx_val)
    idx_test = torch.LongTensor(idx_test)   
    print("Loading idx_test:", idx_test)   
    return add_all, adj, features, labels, idx_train, idx_val, idx_test
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

"""*******************************"""

#def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    """
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
"""

#def preprocess_features(features,adj):
    """Row-normalize feature matrix and convert to tuple representation"""
    #rowsum = np.ones(shape=features.sum(1).shape)
    """
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features) ## type(features) scipy.sparse.csr.csr_matrix'

    return sparse_to_tuple(features)
"""

#def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    """
    adj = sp.coo_matrix(adj)

    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)"""
##  add the node self feature and normalize the primary adjcency matrix


#def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""


    #adj_normalized0 = normalize_adj(adj + sp.eye(adj.shape[0]))
    """
    adj_normalized0 = normalize_adj(adj+sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized0)"""


#def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    """
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})

    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict"""


#def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    """
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
"""
