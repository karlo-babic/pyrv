"""
Model Architecture
==================

Implements the core PyRvNN architecture.
It defines the network's initialization and forward pass functionality, encapsulated within the PyRvNN class.
PyRvNN is the backbone of Pyramidal Recursive Learning (PyRv), a novel text representation learning approach.
It recursively composes (encodes) two input embeddings into a single embedding and trains in an unsupervised manner by learning to decode that embedding.
This module is central to the composition process of the PyRv framework.
"""

from globalvars import *
import tensorflow as tf
import numpy as np


class PyRvNN(tf.keras.Model):
    """
    Pyramidal Recursive Neural Network (PyRvNN) model.

    This model implements a recursive encoding-decoding and autoregressive mechanism for self-supervised learning.
    It encodes two input embeddings into a single latent representation and learns to decode that representation back into its components (autoencoding) and neighboring components (autoregression).
    
    Layers:
        encoder_input_oh (Dense): Processes one-hot encoded token inputs.
        encoder_input_ds (Dense): Processes dense vector embeddings.
        encoder_emb (Dense): Computes the final embedding representation.
        
        output_enc_oh_left (Dense): Decodes left-side one-hot token output.
        output_enc_oh_right (Dense): Decodes right-side one-hot token output.
        output_enc_ds_left (Dense): Decodes left-side dense embedding output.
        output_enc_ds_right (Dense): Decodes right-side dense embedding output.
        
        output_reg_oh_left (Dense): Generates autoregressive left-side one-hot token output.
        output_reg_oh_right (Dense): Generates autoregressive right-side one-hot token output.
        output_reg_ds_left (Dense): Generates autoregressive left-side dense embedding output.
        output_reg_ds_right (Dense): Generates autoregressive right-side dense embedding output.
    """

    def __init__(self):
        super(PyRvNN, self).__init__()
        self.output_dim = (len(VOCAB)*2 + EMBEDDING_SIZE*2) * 2
        activation = tf.keras.layers.LeakyReLU(alpha=0.01)
        regul = tf.keras.regularizers.l2(0.001)
        
        self.encoder_input_oh = tf.keras.layers.Dense(len(VOCAB)*1, activation=activation, kernel_regularizer=regul)
        self.encoder_input_ds = tf.keras.layers.Dense(EMBEDDING_SIZE*1, activation=activation, kernel_regularizer=regul)

        self.encoder_emb   = tf.keras.layers.Dense(EMBEDDING_SIZE, activation=activation, kernel_regularizer=regul)

        self.output_enc_oh_left  = tf.keras.layers.Dense(len(VOCAB), activation='softmax', kernel_regularizer=regul)
        self.output_enc_oh_right = tf.keras.layers.Dense(len(VOCAB), activation='softmax', kernel_regularizer=regul)
        self.output_enc_ds_left  = tf.keras.layers.Dense(EMBEDDING_SIZE, activation=activation, kernel_regularizer=regul)
        self.output_enc_ds_right = tf.keras.layers.Dense(EMBEDDING_SIZE, activation=activation, kernel_regularizer=regul)
        
        self.output_reg_oh_left  = tf.keras.layers.Dense(len(VOCAB), activation='softmax', kernel_regularizer=regul)
        self.output_reg_oh_right = tf.keras.layers.Dense(len(VOCAB), activation='softmax', kernel_regularizer=regul)
        self.output_reg_ds_left  = tf.keras.layers.Dense(EMBEDDING_SIZE, activation=activation, kernel_regularizer=regul)
        self.output_reg_ds_right = tf.keras.layers.Dense(EMBEDDING_SIZE, activation=activation, kernel_regularizer=regul)

        #      enc     reg
        #   | OH DS | OH DS |

    def encode(self, inputs):
        """
        Encodes two input embeddings into a single latent representation.

        The inputs consist of both one-hot encoded tokens and dense embeddings.
        One-hot representations are used at the bottom level of the pyramid, while dense embeddings are recursively composed into new embeddings.

        Args:
            inputs (Tensor): A tensor of shape (batch_size, 5000), where batch_size is the number of input pairs.

        Returns:
            Tensor: A tensor of shape (batch_size, EMBEDDING_SIZE), representing the learned embeddings.
        """

        inputs_onehot_left  = inputs[:,:len(VOCAB)]
        inputs_onehot_right = inputs[:,len(VOCAB):len(VOCAB)*2]
        inputs_dense_left  = inputs[:,len(VOCAB)*2:EMBEDDING_SIZE]
        inputs_dense_right = inputs[:,len(VOCAB)*2+EMBEDDING_SIZE:]

        inputs_onehot = tf.keras.layers.Concatenate()([inputs_onehot_left, inputs_onehot_right])
        inputs_dense = tf.keras.layers.Concatenate()([inputs_dense_left, inputs_dense_right])
        x_oh = self.encoder_input_oh(inputs_onehot)
        x_ds = self.encoder_input_ds(inputs_dense)

        x = tf.keras.layers.Concatenate()([x_oh, x_ds])
        embs = self.encoder_emb(x)

        return embs

    def decode(self, embs):
        """
        Decodes an embedding into its original components.

        The decoding process includes both autoencoding (reconstructing the original input) and autoregression (predicting neighboring representations).

        Args:
            embs (Tensor): A tensor of shape (batch_size, EMBEDDING_SIZE), representing the encoded embeddings.

        Returns:
            dict: A dictionary containing:
                - "enc": Autoencoding outputs.
                - "reg": Autoregressive outputs.
        """

        output_enc = self.decode_enc(embs)
        output_reg = self.decode_reg(embs)
        return {"enc": output_enc, "reg": output_reg}
    
    def decode_enc(self, embs):
        """
        Decodes an embedding into its original one-hot and dense components.

        This function reconstructs the input representations that were originally encoded into `embs`.

        Args:
            embs (Tensor): A tensor of shape (batch_size, EMBEDDING_SIZE).

        Returns:
            dict: A dictionary containing:
                - "onehot_left": Decoded left-side one-hot representation.
                - "onehot_right": Decoded right-side one-hot representation.
                - "dense_left": Decoded left-side dense embedding.
                - "dense_right": Decoded right-side dense embedding.
        """

        output_enc_oh_left  = self.output_enc_oh_left(embs)
        output_enc_oh_right = self.output_enc_oh_right(embs)
        output_enc_ds_left  = self.output_enc_ds_left(embs)
        output_enc_ds_right = self.output_enc_ds_right(embs)
        
        return {"onehot_left" : output_enc_oh_left,
                "onehot_right": output_enc_oh_right,
                "dense_left"  : output_enc_ds_left,
                "dense_right" : output_enc_ds_right}
                    
    def decode_reg(self, embs):
        """
        Performs autoregressive decoding.

        This function predicts potential neighboring representations based on the learned embedding.

        Args:
            embs (Tensor): A tensor of shape (batch_size, EMBEDDING_SIZE).

        Returns:
            dict: A dictionary containing:
                - "onehot_left": Predicted left-side one-hot representation.
                - "onehot_right": Predicted right-side one-hot representation.
                - "dense_left": Predicted left-side dense embedding.
                - "dense_right": Predicted right-side dense embedding.
        """

        output_reg_oh_left  = self.output_reg_oh_left(embs)
        output_reg_oh_right = self.output_reg_oh_right(embs)
        output_reg_ds_left  = self.output_reg_ds_left(embs)
        output_reg_ds_right = self.output_reg_ds_right(embs)
        
        return {"onehot_left" : output_reg_oh_left,
                "onehot_right": output_reg_oh_right,
                "dense_left"  : output_reg_ds_left,
                "dense_right" : output_reg_ds_right}
    
    def call(self, inputs, training=False, return_embs=False):
        """
        Forward pass of the PyRvNN model.

        This function first encodes the input into an embedding and then decodes it.
        If `return_embs` is set to True, the function also returns the learned embeddings.

        Args:
            inputs (Tensor): A tensor of shape (batch_size, 5000), where batch_size is the number of input pairs.
            training (bool, optional): Indicates whether the model is in training mode. Default is False.
            return_embs (bool, optional): If True, returns the computed embeddings along with the outputs. Default is False.

        Returns:
            dict or tuple:
                - If `return_embs` is False: A dictionary containing encoded and autoregressive outputs.
                - If `return_embs` is True: A tuple (outputs, embeddings), where outputs is the standard output dictionary.
        """

        embs = self.encode(inputs)
        outputs = self.decode(embs)
        
        if return_embs:
            return outputs, embs
        else:
            return outputs
