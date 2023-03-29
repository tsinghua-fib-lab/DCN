# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import tensorflow as tf
from reco_utils.recommender.deeprec.models.sequential.sequential_base_model_contrastive import (
    SequentialBaseModel_Contrastive,
)
from tensorflow.nn import dynamic_rnn
from reco_utils.recommender.deeprec.models.sequential.rnn_cell_implement import (
    Time4LSTMCell,
)

__all__ = ["SLI_RECModel"]


class DUALSLI_RECModel(SequentialBaseModel_Contrastive):
    """SLI Rec model

    Z. Yu, J. Lian, A. Mahmoody, G. Liu and X. Xie, "Adaptive User Modeling with
    Long and Short-Term Preferences for Personailzed Recommendation", in Proceedings of 
    the 28th International Joint Conferences on Artificial Intelligence, IJCAI’19, 
    Pages 4213-4219, AAAI Press, 2019.
    """
    # self.model_name == "SLIRec"
    def _encoder(self, hist_input, target_embed, mask, domain):
        hparams = self.hparams
        with tf.variable_scope(domain):
            # hist_input = self.item_history_embedding
            # mask = self.iterator.mask
            sequence_length = tf.reduce_sum(mask, 1)

            with tf.variable_scope("long_term_asvd"):
                att_outputs1 = self._attention(hist_input, hparams.attention_size)
                att_fea1 = tf.reduce_sum(att_outputs1, 1)
                tf.summary.histogram("att_fea1", att_fea1)

            
            with tf.variable_scope("rnn"):
                rnn_outputs, final_state = dynamic_rnn(
                    Time4LSTMCell(hparams.hidden_size),
                    inputs=hist_input,
                    sequence_length=sequence_length,
                    dtype=tf.float32,
                    scope="time4lstm",
                )
                tf.summary.histogram("rnn_outputs", rnn_outputs)

            with tf.variable_scope("attention_fcn"):
                att_outputs2 = self._attention_fcn(
                    target_embed, rnn_outputs, mask
                )
                att_fea2 = tf.reduce_sum(att_outputs2, 1)
                tf.summary.histogram("att_fea2", att_fea2)

            # ensemble
            with tf.name_scope("alpha"):
   
                user_embed = att_fea1 * hparams.manual_alpha_value + att_fea2 * (1.0 - hparams.manual_alpha_value)
            # model_output = tf.concat([user_embed, self.target_item_embedding], 1)
            # tf.summary.histogram("model_output", model_output)
            return user_embed, att_fea2

    def _build_seq_graph(self):
        """The main function to create sli_rec model.
        
        Returns:
            obj:the output of sli_rec section.
        """
        hist_embedding_mean_i, long_i = self._encoder(self.item_history_embedding, self.target_item_embedding, self.iterator.mask, "i")
        hist_embedding_mean_u, long_u = self._encoder(self.user_history_embedding, self.target_user_embedding, self.iterator.mask_u, "u")

        model_output_i = tf.concat([self.target_item_embedding, hist_embedding_mean_i], -1)
        model_output_u = tf.concat([self.target_user_embedding, hist_embedding_mean_u], -1)
        model_output_u_i = tf.concat([self.target_user_embedding, self.target_item_embedding], -1)
        user_rep = tf.layers.dense(hist_embedding_mean_i, self.hidden_size, activation=None)
        item_rep = tf.layers.dense(hist_embedding_mean_u, self.hidden_size, activation=None)
        long_u = tf.layers.dense(long_u, self.hidden_size, activation=None)
        long_i = tf.layers.dense(long_i, self.hidden_size, activation=None)
        #  import pdb; pdb.set_trace()
        #  model_output = tf.concat([self.target_item_embedding, self.hist_embedding_concat], -1)
        #  model_output = self.hist_embedding_concat
        # Inner Product
        #  model_output = tf.reduce_sum(self.target_item_embedding * self.recent_embedding_mean, axis=-1)
        return user_rep, item_rep, model_output_u, model_output_i, model_output_u_i, long_u, long_i

    def _attention_fcn(self, query, user_embedding, mask, return_alpha=False):
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
            boolean_mask = tf.equal(mask, tf.ones_like(mask))

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
