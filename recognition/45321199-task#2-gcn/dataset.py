"""
    This file contains the class "Dataloader", which handles retrieving and
    categorising the data.
"""

__author__ = "Lachlan Comino"

import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import LabelBinarizer as lb
import tensorflow as tf
from tensorflow.keras import utils


class DataLoader:
    """
        Class to hold all functionality for processing the data.
    """
    def download_data(self):
        """
            Execute data retrieval.
        """
        data_url = "https://graphmining.ai/datasets/ptg/facebook.npz"
        data_fname = "facebook.npz"

        self.data_dir = utils.get_file(data_fname, data_url)
        print(f"Data saved at {self.data_dir}")

    def process_data(self):
        """
            Categorises the data so it's usable. Initialises a dictionary with
            all the segments of data.
        """
        # Unpack data into components
        with np.load(self.data_dir) as data:
            edges, features, labels = data.values()

        label_types = np.unique(labels)
        len_label_types = len(label_types)
        len_vertices = features.shape[0]
        len_features = features.shape[1]
        len_edges = len(edges) // 2

        # Print dimensions of data
        print(
            f"Label Types:\t{label_types}",
            f"Edges:      \t{len_edges}",
            f"Vertices:   \t{len_vertices}",
            f"Features:   \t{len_features}",
            sep='\n'
        )

        # Normalise Features
        features /= features.sum(1).reshape(-1, 1)

        # get blank adjacency matrix representation of graph
        graph_adj_mat = self.adj_mat_repr(len_vertices, edges)

        # one-hot encode labels i.e. 0,1,2,3 -> 1000,0100,0010,0001
        encoder = lb()
        encoded_labels = encoder.fit_transform(labels)

        #retrieve range for each subset of the data
        train_range, val_range, test_range = self.split_data(len_vertices)

        # split data into train/validation/test; 0.2:0.2:0.6 ratio
        train_mask = np.zeros(len_vertices, dtype=np.bool)
        val_mask = np.zeros(len_vertices, dtype=np.bool)
        test_mask = np.zeros(len_vertices, dtype=np.bool)

        # Label each vertice with training, validation or test
        train_mask[train_range] = True
        val_mask[val_range] = True
        test_mask[test_range] = True
        
        # make tensors
        tf_features = tf.convert_to_tensor(features)

        validation_data = ([tf_features, graph_adj_mat], encoded_labels, val_mask)

        # package processed data
        self.processed_data = {
            "train_mask" : train_mask,
            "val_mask": val_mask,
            "test_mask": test_mask,
            "encoded_labels" : encoded_labels,
            "label_types" : label_types,
            "len_label_types" : len_label_types,
            "len_vertices" : len_vertices ,
            "len_features" : len_features ,
            "len_edges" : len_edges,
            "validation_data" : validation_data,
            "train_mask" : train_mask,
            "val_mask" : val_mask,
            "test_mask" : test_mask
        }

    def split_data(self, len_vertices):
        """
            @Params:
                len_vertices: amount of features or nodes
            @Returns: 
                The ranges for the different subsets of data.
        """
        train_range = range(len_vertices // 5)
        val_range = range(len_vertices // 5, 2 * len_vertices // 5)
        test_range = range(2 * len_vertices // 5, len_vertices)
        return train_range, val_range, test_range
    
    @staticmethod
    def adj_mat_repr(len_vertices, edges):
        """
            Creates the adjacency matrix representation of the graph.
            @Params:
                len_vertices: amount of features or nodes.
                edges: edge list
            @Returns
                The adjacency matrix as a sparse tensor.
        """
        A = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(len_vertices, len_vertices), dtype=np.float64)

        A_ = A + sp.eye(A.shape[0])
        D = sp.diags(np.power(np.array(A_.sum(1)), -0.5).flatten())
        A_ = D.dot(A_).dot(D).tocoo()
        
        indices = np.mat([A_.row, A_.col]).transpose()

        return tf.SparseTensor(indices, A_.data, A_.shape)

    def load_data(self):
        """
            Runs all processing for the DataLoader
        """
        self.download_data()
        self.process_data()
        return self.processed_data
