# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import tensorflow as tf
from reco_utils.recommender.deeprec.models.sequential.sequential_base_model_contrastive_noLong import (
    SequentialBaseModel_Contrastive_noLong,
)
from tensorflow.nn import dynamic_rnn
from reco_utils.recommender.deeprec.deeprec_utils import load_dict

__all__ = ["DUALDINModel"]


class DUALDINModel(SequentialBaseModel_Contrastive_noLong):

    def _build_seq_graph(self):
        """The main function to create din model.
        
        Returns:
            obj:the output of din section.
        """
        with tf.name_scope('din'):
            hist_input_u = self.user_history_embedding

            hist_input = self.item_history_embedding
            self.mask_u = self.iterator.mask_u
            self.mask = self.iterator.mask
            self.real_mask_u = tf.cast(self.mask_u, tf.float32)

            self.real_mask = tf.cast(self.mask, tf.float32)
            self.hist_embedding_sum_u = tf.reduce_sum(hist_input_u*tf.expand_dims(self.real_mask_u, -1), 1)
            self.hist_embedding_sum = tf.reduce_sum(hist_input*tf.expand_dims(self.real_mask, -1), 1)
            attention_output_u = self._attention_fcn_din(self.target_user_embedding, hist_input_u, 'user')
            att_fea_u = tf.reduce_sum(attention_output_u, 1)
            attention_output = self._attention_fcn_din(self.target_item_embedding, hist_input, 'item')
            att_fea = tf.reduce_sum(attention_output, 1)
            tf.summary.histogram('att_fea', att_fea)

        model_output_u = tf.concat([self.target_user_embedding, att_fea_u, self.hist_embedding_sum_u], -1)
        model_output_i = tf.concat([self.target_item_embedding, att_fea, self.hist_embedding_sum], -1)
        model_output_u_i = tf.concat([self.target_item_embedding, self.target_user_embedding], -1)

        user_rep = tf.layers.dense(self.hist_embedding_sum, self.hidden_size, activation=None)
        item_rep = tf.layers.dense(self.hist_embedding_sum_u, self.hidden_size, activation=None)

        # tf.summary.histogram("model_output", model_output)
        return user_rep, item_rep, model_output_u, model_output_i, model_output_u_i
    def _attention_fcn_din(self, query, user_embedding, name='item', return_alpha=False):
        """Apply attention by fully connected layers.

        Args:
            query (obj): The embedding of target item which is regarded as a query in attention operations.
            user_embedding (obj): The output of RNN layers which is regarded as user modeling.

        Returns:
            obj: Weighted sum of user modeling.
        """
        hparams = self.hparams
        with tf.variable_scope("attention_fcn_%s"%name):
            query_size = query.shape[1].value
            boolean_mask = tf.equal(self.mask, tf.ones_like(self.mask))

            attention_mat = tf.get_variable(
                name="attention_mat",
                shape=[user_embedding.shape.as_list()[-1], query_size],
                initializer=self.initializer,
            )
            att_inputs = tf.tensordot(user_embedding, attention_mat, [[2], [0]])

            queries = tf.reshape(
                tf.tile(query, [1, att_inputs.shape[1].value]), tf.shape(att_inputs)
            )
            last_hidden_nn_layer = tf.concat(
                [att_inputs, queries, att_inputs - queries, att_inputs * queries], -1
            )
            att_fnc_output = self._fcn_net(
                last_hidden_nn_layer, hparams.att_fcn_layer_sizes, scope="att_fcn"
            )
            att_fnc_output = tf.squeeze(att_fnc_output, -1)
            mask_paddings = tf.ones_like(att_fnc_output) * (-(2 ** 32) + 1)
            att_weights = tf.nn.softmax(
                tf.where(boolean_mask, att_fnc_output, mask_paddings),
                name="att_weights",
            )
            output = user_embedding * tf.expand_dims(att_weights, -1)
            if not return_alpha:
                return output
            else:
                return output, att_weights
