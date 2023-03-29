# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import tensorflow as tf
from reco_utils.recommender.deeprec.models.sequential.sequential_base_model_contrastive import (
    SequentialBaseModel_Contrastive,
)
from tensorflow.contrib.rnn import GRUCell
from reco_utils.recommender.deeprec.models.sequential.rnn_cell_implement import (
    VecAttGRUCell,
)
from reco_utils.recommender.deeprec.deeprec_utils import load_dict

from reco_utils.recommender.deeprec.models.sequential.rnn_dien import dynamic_rnn

__all__ = ["DUALDIENModel"]


class DUALDIENModel(SequentialBaseModel_Contrastive):

    def _build_seq_graph(self):
        """The main function to create din model.
        
        Returns:
            obj:the output of din section.
        """
        hparams = self.hparams
        with tf.name_scope('dualdien'):
            hist_input_u = self.user_history_embedding

            hist_input = self.item_history_embedding
            self.mask_u = self.iterator.mask_u

            self.mask = self.iterator.mask
            self.real_mask_u = tf.cast(self.mask_u, tf.float32)

            self.real_mask = tf.cast(self.mask, tf.float32)
            self.hist_embedding_sum_u = tf.reduce_sum(hist_input_u*tf.expand_dims(self.real_mask_u, -1), 1)

            self.hist_embedding_sum = tf.reduce_sum(hist_input*tf.expand_dims(self.real_mask, -1), 1)
            with tf.name_scope('rnn_1'):
                # self.mask = self.iterator.mask
                self.sequence_length = tf.reduce_sum(self.mask, 1)
                rnn_outputs, _ = dynamic_rnn(
                    GRUCell(hparams.hidden_size),
                    inputs=hist_input,
                    sequence_length=self.sequence_length,
                    dtype=tf.float32,
                    scope="gru1"
                )
                # tf.summary.histogram('GRU_outputs', rnn_outputs)        
            with tf.name_scope('rnn_1_user'):
                # self.mask = self.iterator.mask
                self.sequence_length_u = tf.reduce_sum(self.mask_u, 1)
                rnn_outputs_u, _ = dynamic_rnn(
                    GRUCell(hparams.hidden_size),
                    inputs=hist_input_u,
                    sequence_length=self.sequence_length_u,
                    dtype=tf.float32,
                    scope="gru1_user"
                )
                # tf.summary.histogram('GRU_outputs', rnn_outputs)        

            # Attention layer
            with tf.variable_scope('Attention_layer_1'):
                _, alphas = self._attention_fcn(self.target_item_embedding, rnn_outputs, return_alpha=True)
            with tf.variable_scope('Attention_layer_1_user'):
                _, alphas_u = self._attention_fcn(self.target_user_embedding, rnn_outputs_u, return_alpha=True)

            with tf.name_scope('rnn_2'):
                _, final_state = dynamic_rnn(
                    VecAttGRUCell(hparams.hidden_size),
                    inputs=rnn_outputs,
                    att_scores = tf.expand_dims(alphas, -1),
                    sequence_length=self.sequence_length,
                    dtype=tf.float32,
                    scope="gru2"
                )
                # tf.summary.histogram('GRU2_Final_State', final_state)
            with tf.name_scope('rnn_2_user'):
                _, final_state_u = dynamic_rnn(
                    VecAttGRUCell(hparams.hidden_size),
                    inputs=rnn_outputs_u,
                    att_scores = tf.expand_dims(alphas_u, -1),
                    sequence_length=self.sequence_length_u,
                    dtype=tf.float32,
                    scope="gru2_user"
                )
        model_output_u = tf.concat([self.target_user_embedding, final_state_u, self.hist_embedding_sum_u, self.target_user_embedding*self.hist_embedding_sum_u], 1)

        model_output_i = tf.concat([self.target_item_embedding, final_state, self.hist_embedding_sum, self.target_item_embedding*self.hist_embedding_sum], 1)
        model_output_u_i = tf.concat([self.target_item_embedding, self.target_user_embedding], 1)
        user_rep = tf.layers.dense(final_state, self.hidden_size, activation=None)
        item_rep = tf.layers.dense(final_state_u, self.hidden_size, activation=None)
        
        long_u = tf.layers.dense(self.hist_embedding_sum_u, self.hidden_size, activation=None)
        long_i = tf.layers.dense(self.hist_embedding_sum, self.hidden_size, activation=None)
        # tf.summary.histogram("model_output", model_output)
        
        return user_rep, item_rep, model_output_u, model_output_i, model_output_u_i, long_u, long_i

    def _attention_fcn(self, query, user_embedding, return_alpha=False):
        """Apply attention by fully connected layers.

        Args:
            query (obj): The embedding of target item which is regarded as a query in attention operations.
            user_embedding (obj): The output of RNN layers which is regarded as user modeling.

        Returns:
            obj: Weighted sum of user modeling.
        """
        hparams = self.hparams
        with tf.variable_scope("attention_fcn"):
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
