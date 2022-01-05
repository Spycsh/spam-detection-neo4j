from data_loader import DataLoader
from GAS_model import GASModel
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras import optimizers

from typing import Tuple, Union
import scipy.sparse as sp


def sparse_to_tuple(sparse_mx: sp.coo_matrix) -> Tuple[np.array, np.array,
                                                       np.array]:
    """
    Convert sparse matrix to tuple representation.
    :param sparse_mx: the graph adjacency matrix in scipy sparse matrix format
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


def preprocess_feature(features: np.array, to_tuple: bool = True) -> \
        Union[Tuple[np.array, np.array, np.array], sp.csr_matrix]:
    """
    Row-normalize feature matrix and convert to tuple representation
    Parts of this code file were originally forked from
    https://github.com/tkipf/gcn
    :param features: the node feature matrix
    :param to_tuple: whether cast the feature matrix to scipy sparse tuple
    """
    features = sp.lil_matrix(features)
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)

    if to_tuple:
        return sparse_to_tuple(features)
    else:
        return features

def preprocess_adj(adj: np.array, to_tuple: bool = True) -> \
        Union[Tuple[np.array, np.array, np.array], sp.coo_matrix]:
    """
    Preprocessing of adjacency matrix for simple GCN model
    and conversion to tuple representation.
    Parts of this code file were originally forked from
    https://github.com/tkipf/gcn
    :param adj: the graph adjacency matrix
    """
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))

    if to_tuple:
        return sparse_to_tuple(adj_normalized)
    else:
        return

def normalize_adj(adj: np.array) -> sp.coo_matrix:
    """
    Symmetrically normalize adjacency matrix
    Parts of this code file were originally forked from
    https://github.com/tkipf/gcn
    :param adj: the graph adjacency matrix
    """
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


if __name__ == "__main__":
    # declare the super-parameters here
    lr = 0.001
    epochs = 30
    seed = 123
    batch_size = 1000
    momentum = 0.9

    dataLoader = DataLoader("bolt://localhost:7687", "neo4j", "admin")
    adj_list, features, [X_train, X_test], y = dataLoader.obtain_graph()

    label = tf.convert_to_tensor(y, dtype=tf.float32)

    # r_support
    homo_graph = np.array(adj_list[6], dtype=float)

    # r_feature
    review_vectors = np.array(features[0], dtype=float)

    # convert to dense tensors
    features[0] = tf.convert_to_tensor(features[0], dtype=tf.float32)
    features[1] = tf.convert_to_tensor(features[1], dtype=tf.float32)
    features[2] = tf.convert_to_tensor(features[2], dtype=tf.float32)


    # get sparse tuples
    r_feature = preprocess_feature(review_vectors)
    r_support = preprocess_adj(homo_graph)


    kwargs={}
    # initialize the model parameters
    kwargs['reviews_num'] = features[0].shape[0]
    # args.class_size = y.shape[1]
    kwargs['input_dim_i'] = features[2].shape[1]
    kwargs['input_dim_u'] = features[1].shape[1]
    kwargs['input_dim_r'] = features[0].shape[1]
    # 5
    kwargs['input_dim_u_x'] = features[1].shape[0]
    # 3
    kwargs['input_dim_i_x'] = features[2].shape[0]

    kwargs['input_dim_r_gcn'] = r_feature[2][1]
    kwargs['num_features_nonzero'] = r_feature[1].shape
    kwargs['h_u_size'] = adj_list[0].shape[1] * (
            kwargs.get('input_dim_r') + kwargs.get('input_dim_i'))
    kwargs['h_i_size'] = adj_list[2].shape[1] * (
            kwargs['input_dim_r'] + kwargs['input_dim_u'])


    # get sparse tensors
    r_feature = tf.SparseTensor(*r_feature)
    r_support = [tf.cast(tf.SparseTensor(*r_support), dtype=tf.float32)]
    # masks = [X_train, X_test]



    model = GASModel(kwargs)
    optimizer = optimizers.Adam(lr)

    # train
    for _ in tqdm(range(epochs)):
        with tf.GradientTape() as tape:
            train_loss, train_acc = model(
                [adj_list, r_support, features, r_feature, label, X_train], ) # pass parameters to call function in the model
            print(f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}")

        grads = tape.gradient(train_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # test
    test_loss, test_acc = model(
        [adj_list, r_support, features, r_feature, label, X_test],) # pass parameters to call function in the model

    print(f"test_loss: {test_loss:.4f}, test_acc: {test_acc:.4f}")

