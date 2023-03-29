# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import tensorflow as tf
from reco_utils.recommender.deeprec.models.sequential.sequential_base_model_contrastive_noLong import (
    SequentialBaseModel_Contrastive_noLong,
)

__all__ = ["DUALCaserModel"]


class DUALCaserModel(SequentialBaseModel_Contrastive_noLong):
    """Caser Model

    J. Tang and K. Wang, "Personalized top-n sequential recommendation via convolutional 
    sequence embedding", in Proceedings of the Eleventh ACM International Conference on 
    Web Search and Data Mining, ACM, 2018.
    """

    def __init__(self, hparams, iterator_creator):
        """Initialization of variables for caser 

        Args:
            hparams (obj): A tf.contrib.training.HParams object, hold the entire set of hyperparameters.
            iterator_creator (obj): An iterator to load the data.
        """
        self.hparams = hparams
        self.L = hparams.L  # history sequence that involved in convolution shape
        self.T = hparams.T  # prediction shape
        self.n_v = hparams.n_v  # number of vertical convolution layers
        self.n_h = hparams.n_h  # number of horizonal convolution layers
        #                                 T=1, n_v=128, n_h=128, L=3,

        self.lengths = [
            i + 1 for i in range(self.L)
        ]  # horizonal convolution filter shape
        super().__init__(hparams, iterator_creator)

    def _build_seq_graph(self):
        """The main function to create caser model.
        
        Returns:
            obj:the output of caser section.
        """
        with tf.variable_scope("dualcaser"):
            cnn_output_u = self._caser_cnn(self.user_history_embedding, self.user_embedding_dim, "user")
            cnn_output_i = self._caser_cnn(self.item_history_embedding, self.item_embedding_dim, "item")
            model_output_u = tf.concat([cnn_output_u, self.target_user_embedding], 1)
            model_output_i = tf.concat([cnn_output_i, self.target_item_embedding], 1)
            model_output_u_i = tf.concat([self.target_user_embedding, self.target_item_embedding], 1)

            user_rep = tf.layers.dense(cnn_output_i, self.hidden_size, activation=None)
            item_rep = tf.layers.dense(cnn_output_u, self.hidden_size, activation=None)
      
            # tf.summary.histogram("model_output", model_output)
            return user_rep, item_rep, model_output_u, model_output_i, model_output_u_i

    def _add_cnn(self, hist_matrix, vertical_dim, scope):
        """The main function to use CNN at both vertical and horizonal aspects.
        
        Args:
            hist_matrix (obj): The output of history sequential embeddings
            vertical_dim (int): The shape of embeddings of input
            scope (obj): The scope of CNN input.

        Returns:
            obj:the output of CNN layers.
        """
        with tf.variable_scope(scope):
            with tf.variable_scope("vertical"):
                embedding_T = tf.transpose(hist_matrix, [0, 2, 1])
                out_v = self._build_cnn(embedding_T, self.n_v, vertical_dim)
                out_v = tf.layers.flatten(out_v)
            with tf.variable_scope("horizonal"):
                out_hs = []
                for h in self.lengths:
                    conv_out = self._build_cnn(hist_matrix, self.n_h, h)
                    max_pool_out = tf.reduce_max(
                        conv_out, reduction_indices=[1], name="max_pool_{0}".format(h)
                    )
                    out_hs.append(max_pool_out)
                out_h = tf.concat(out_hs, 1)
        return tf.concat([out_v, out_h], 1)

    def _caser_cnn(self, his_embedding, his_dim, name = "item"):
        """The main function to use CNN at both item and category aspects.
        
        Returns:
            obj:the concatenated output of two parts of item and catrgory.
        """
        his_out = self._add_cnn(
            his_embedding, his_dim, name
        )
        # tf.summary.histogram("his_out", his_out)
        
        cnn_output = his_out
        # tf.summary.histogram("cnn_output", cnn_output)
        return cnn_output

    def _build_cnn(self, history_matrix, nums, shape):
        """Call a CNN layer.
        
        Returns:
            obj:the output of cnn section.
        """
        return tf.layers.conv1d(
            history_matrix,
            nums,
            shape,
            activation=tf.nn.relu,
            name="conv_" + str(shape),
        )
