# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import tensorflow as tf
from reco_utils.recommender.deeprec.models.sequential.sequential_base_model_contrastive_noLong import (
    SequentialBaseModel_Contrastive_noLong,
)
from tensorflow.contrib.rnn import GRUCell, LSTMCell
from tensorflow.nn import dynamic_rnn

__all__ = ["GRU4RecModel"]


class DualGRU4RecModel(SequentialBaseModel_Contrastive_noLong):
    """GRU4Rec Model

    B. Hidasi, A. Karatzoglou, L. Baltrunas, D. Tikk, "Session-based Recommendations 
    with Recurrent Neural Networks", ICLR (Poster), 2016.
    """
    # def __init__(self):

    def _build_seq_graph(self):
        """The main function to create GRU4Rec model.
        
        Returns:
            obj:the output of GRU4Rec section.
        """
        with tf.variable_scope("gru4rec"):
            # final_state = self._build_lstm()
            final_state_user, final_state = self._build_gru()
            model_output_u = tf.concat([final_state_user, self.target_user_embedding], 1)

            model_output_i = tf.concat([final_state, self.target_item_embedding], 1)

            model_output_u_i = tf.concat([self.target_item_embedding, self.target_user_embedding], 1) 
            # model_output_u_i = tf.concat([self.target_item_embedding, self.target_user_embedding], 1) 

            # tf.summary.histogram("model_output", model_output)
            user_rep = tf.layers.dense(final_state, self.hidden_size, activation=None)
            item_rep = tf.layers.dense(final_state_user, self.hidden_size, activation=None)
            # item_target_rep = tf.layers.dense(self.target_item_embedding, self.hidden_size, activation=None)
            # user_target_rep = tf.layers.dense(self.target_user_embedding, self.hidden_size, activation=None)
            # attention_output_u_rep = tf.layers.dense(attention_output_u, self.hidden_size, activation=None)
            # attention_output_i_rep = tf.layers.dense(attention_output_i, self.hidden_size, activation=None)
            return user_rep, item_rep, model_output_u, model_output_i, model_output_u_i

    def _build_lstm(self):
        """Apply an LSTM for modeling.

        Returns:
            obj: The output of LSTM section.
        """
        with tf.name_scope("lstm"):
            self.mask = self.iterator.mask
            self.sequence_length = tf.reduce_sum(self.mask, 1)
            self.history_embedding = self.item_history_embedding
            print(self.hidden_size)
            rnn_outputs, final_state = dynamic_rnn(
                LSTMCell(self.hidden_size),
                inputs=self.history_embedding,
                sequence_length=self.sequence_length,
                dtype=tf.float32,
                scope="lstm",
            )
            # tf.summary.histogram("LSTM_outputs", rnn_outputs)
            return final_state[1]

    def _build_gru(self):
        """Apply a GRU for modeling.

        Returns:
            obj: The output of GRU section.
        """
        with tf.name_scope("gru"):
            self.mask = self.iterator.mask
            self.mask_u = self.iterator.mask_u
            self.real_mask = tf.cast(self.mask, tf.float32)
            self.real_mask_u = tf.cast(self.mask_u, tf.float32)

            self.sequence_length = tf.reduce_sum(self.mask, 1)
            self.sequence_length_u = tf.reduce_sum(self.mask_u, 1)

            self.history_embedding = self.item_history_embedding
            self.history_user_embedding = self.user_history_embedding
            # attention_output_u, alphas = self._attention_fcn(self.target_user_embedding, self.history_user_embedding, 'Att_u', False, self.iterator.mask, return_alpha=True)
            # attention_output_i, alphas = self._attention_fcn(self.target_item_embedding, self.history_embedding, 'Att_i', False, self.iterator.mask, return_alpha=True)
            rnn_outputs_user, final_state_user = dynamic_rnn(
                GRUCell(self.hidden_size),
                inputs=self.history_user_embedding,
                sequence_length=self.sequence_length_u,
                dtype=tf.float32,
                scope="gru_u",
            )
            # _, final_state_user_c = dynamic_rnn(
            #     GRUCell(self.hidden_size),
            #     inputs=self.history_user_embedding,
            #     sequence_length=self.sequence_length_u,
            #     dtype=tf.float32,
            #     scope="gru_u_c",
            # )
            # print(self.history_embedding.shape.as_list() )
            rnn_outputs, final_state = dynamic_rnn(
                GRUCell(self.hidden_size),
                inputs=self.history_embedding,
                sequence_length=self.sequence_length,
                dtype=tf.float32,
                scope="gru",
            )
            # _, final_state_c = dynamic_rnn(
            #     GRUCell(self.hidden_size),
            #     inputs=self.history_embedding,
            #     sequence_length=self.sequence_length,
            #     dtype=tf.float32,
            #     scope="gru_c",
            # )

            # tf.summary.histogram("GRU_outputs", rnn_outputs)
            # rnn_outputs_mean = tf.reduce_sum(rnn_outputs*tf.expand_dims(self.real_mask, -1), 1)/tf.reduce_sum(self.real_mask, 1, keepdims=True)
            # rnn_outputs_user_mean = tf.reduce_sum(rnn_outputs_user*tf.expand_dims(self.real_mask_u, -1), 1)/tf.reduce_sum(self.real_mask_u, 1, keepdims=True)

            return final_state_user, final_state
