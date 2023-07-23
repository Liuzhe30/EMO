# Enformer structure for comparison

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Layer

from .utils.Transformer import MultiHeadSelfAttention
from .utils.Transformer import TransformerBlock
from .utils.Transformer import TokenAndPositionEmbedding

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    # add extra dimensions to add the padding
    # to the attention logits.
    return  seq[:, tf.newaxis, tf.newaxis, :]# (batch_size, 1, 1, seq_len)

# generate by ChatGPT-4
class AttentionPooling1D(Layer):
    def __init__(self, pool_size=2, **kwargs):
        super(AttentionPooling1D, self).__init__(**kwargs)
        self.pool_size = pool_size

    def build(self, input_shape):
        self.filters = input_shape[-1]
        self.attention_weights = self.add_weight(name='attention_weights',
                                                 shape=(self.filters,),
                                                 initializer='ones',
                                                 trainable=True)
        super(AttentionPooling1D, self).build(input_shape)

    def call(self, inputs):
        pool_size = self.pool_size
        attention_weights = self.attention_weights / tf.reduce_sum(self.attention_weights)
        inputs *= tf.expand_dims(attention_weights, axis=0)
        output = tf.nn.max_pool1d(inputs, ksize=pool_size,
                                  strides=pool_size, padding='VALID')
        return output

    def get_config(self):
        config = super(AttentionPooling1D, self).get_config()
        config.update({'pool_size': self.pool_size})
        return config

def build_EFM_small(): # no attention pooling

    # hyper-paramaters
    maxlen = 1000
    window_len = 51
    variant_box_len = 14
    block_size = 20
    vocab_size = 5
    embed_dim = 64
    num_heads = 8
    ff_dim = 256
    pos_embed_dim = 64
    seq_embed_dim_mut = 55
    seq_embed_dim_bet = 32

    ####### inputs (from data generator)
    input1 = Input(shape=(window_len, 4), name = 'input_before_51') # 51 seq-before-mutation
    input2 = Input(shape=(window_len, 4), name = 'input_after_51') # 51 seq-after-mutation
    input3 = Input(shape=(window_len,), name = 'input_atac_51') # 51 atac

    input4 = Input(shape=(maxlen, 4), name = 'input_bet_seq') # maxlen variant-tss-between-seq
    input5 = Input(shape=(maxlen,), name = 'input_atac_bet') # maxlen variant-tss-between-atac

    input6 = Input(shape=(window_len,), name = 'input_mask1') # mask-seq-51
    input7 = Input(shape=(maxlen,), name = 'input_mask2') # mask-seq-between

    ####### merge inputs of the same scale
    new_input3 = layers.Reshape((window_len,1), name = 'reshape_input_51')(input3)
    input_mut = layers.concatenate([input1, input2, new_input3], axis=-1)
    #print('input_mut.get_shape()', input_mut.get_shape()) # (None, 51, 9)
    new_input5 = layers.Reshape((maxlen,1), name = 'reshape_input_between')(input5)
    input_between = layers.concatenate([input4, new_input5], axis=-1)
    #print('input_between.get_shape()', input_between.get_shape()) # (None, maxlen, 5)

    # -----init all used basic layers-----
    leaky_relu = tf.keras.layers.LeakyReLU()
    embedding_layer_mut = TokenAndPositionEmbedding(window_len, vocab_size, embed_dim, pos_embed_dim, seq_embed_dim_mut)
    embedding_layer_bet = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim, pos_embed_dim, seq_embed_dim_bet)
    trans_block_mut1 = TransformerBlock(embed_dim, num_heads, ff_dim)
    trans_block_bet1 = TransformerBlock(embed_dim, num_heads, ff_dim)
    trans_block_mut2 = TransformerBlock(embed_dim, num_heads, ff_dim)
    trans_block_bet2 = TransformerBlock(embed_dim, num_heads, ff_dim)
    
    # 4 conv-resnet layers
    bet_cnn = tf.keras.layers.Conv1D(3, 32, kernel_initializer='he_uniform', padding='same')(input_between)
    bet_cnn = tf.keras.layers.Conv1D(5, 64, kernel_initializer='he_uniform', padding='same')(bet_cnn)
    merge = tf.keras.layers.Concatenate(axis=-1)([input_between, bet_cnn])
    #print('merge.get_shape()', merge.get_shape()) # (None, 1000, 10)
    bet_cnn = tf.keras.layers.Conv1D(16, 64, kernel_initializer='he_uniform', padding='same')(merge)
    bet_cnn = tf.keras.layers.Conv1D(22, 64, kernel_initializer='he_uniform', padding='same')(bet_cnn)
    bet_cnn = tf.keras.layers.Concatenate(axis=-1)([merge, bet_cnn])
    #print('bet_cnn.get_shape()', bet_cnn.get_shape()) # (None, 1000, 32)
    
    bet_cnn_mask = tf.keras.layers.Reshape((-1,))(input7)
    #print('bet_cnn_mask.get_shape()', bet_cnn_mask.get_shape())
    mut_mask = create_padding_mask(input6)
    bet_mask = create_padding_mask(bet_cnn_mask)

    mut = embedding_layer_mut([input6,input_mut])
    bet = embedding_layer_bet([bet_cnn_mask,bet_cnn])
    #print('embedding_layer_mut.get_shape()', mut.get_shape()) # (None, 51, 64)
    #print('embedding_layer_bet.get_shape()', bet.get_shape()) # (None, 1000, 64)

    mut = trans_block_mut1(mut, mut_mask)
    bet = trans_block_bet1(bet, bet_mask)
    mut = trans_block_mut2(mut, mut_mask)
    bet = trans_block_bet2(bet, bet_mask)

    mut_bew = tf.keras.layers.Concatenate(axis=1)([mut, bet])
    mut_bew = tf.keras.layers.Conv1D(4, 1, kernel_initializer='he_uniform')(mut_bew)
    mut_bew = tf.keras.layers.GlobalAveragePooling1D()(mut_bew)
    output = tf.keras.layers.Dense(2, activation = 'softmax', name = 'output_softmax')(mut_bew)

    model = Model(inputs=[input1, input2, input3, input4, input5, input6, input7], outputs=output)
    return model


def build_EFM_middle(): # with average pooling
    # hyper-paramaters
    maxlen = 10000
    window_len = 51
    variant_box_len = 14
    block_size = 20
    vocab_size = 5
    embed_dim = 128
    num_heads = 8
    ff_dim = 256
    pos_embed_dim = 128
    seq_embed_dim_mut = 55+64
    seq_embed_dim_bet = 64

    ####### inputs (from data generator)
    input1 = Input(shape=(window_len, 4), name = 'input_before_51') # 51 seq-before-mutation
    input2 = Input(shape=(window_len, 4), name = 'input_after_51') # 51 seq-after-mutation
    input3 = Input(shape=(window_len,), name = 'input_atac_51') # 51 atac

    input4 = Input(shape=(maxlen, 4), name = 'input_bet_seq') # maxlen variant-tss-between-seq
    input5 = Input(shape=(maxlen,), name = 'input_atac_bet') # maxlen variant-tss-between-atac

    input6 = Input(shape=(window_len,), name = 'input_mask1') # mask-seq-51
    input7 = Input(shape=(maxlen,), name = 'input_mask2') # mask-seq-between

    ####### merge inputs of the same scale
    new_input3 = layers.Reshape((window_len,1), name = 'reshape_input_51')(input3)
    input_mut = layers.concatenate([input1, input2, new_input3], axis=-1)
    #print('input_mut.get_shape()', input_mut.get_shape()) # (None, 51, 9)
    new_input5 = layers.Reshape((maxlen,1), name = 'reshape_input_between')(input5)
    input_between = layers.concatenate([input4, new_input5], axis=-1)
    #print('input_between.get_shape()', input_between.get_shape()) # (None, maxlen, 5)

    # -----init all used basic layers-----
    leaky_relu = tf.keras.layers.LeakyReLU()
    embedding_layer_mut = TokenAndPositionEmbedding(window_len, vocab_size, embed_dim, pos_embed_dim, seq_embed_dim_mut)
    embedding_layer_bet = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim, pos_embed_dim, seq_embed_dim_bet)
    trans_block_mut1 = TransformerBlock(embed_dim, num_heads, ff_dim)
    trans_block_bet1 = TransformerBlock(embed_dim, num_heads, ff_dim)
    trans_block_mut2 = TransformerBlock(embed_dim, num_heads, ff_dim)
    trans_block_bet2 = TransformerBlock(embed_dim, num_heads, ff_dim)
    
    # 4 conv-resnet layers
    # This Resnet block significantly reduces the fitting ability of the model: cannot fit
    bet_cnn = tf.keras.layers.Conv1D(3, 16, kernel_initializer='he_uniform', padding='same')(input_between)
    bet_cnn = tf.keras.layers.Conv1D(9, 16, kernel_initializer='he_uniform', padding='same')(bet_cnn)
    #merge = tf.keras.layers.Concatenate(axis=-1)([input_between, bet_cnn])
    bet_cnn = tf.keras.layers.Conv1D(16, 32, kernel_initializer='he_uniform', padding='same')(bet_cnn)
    bet_cnn = tf.keras.layers.Conv1D(64, 128, kernel_initializer='he_uniform', padding='same')(bet_cnn)
    #bet_cnn = tf.keras.layers.Concatenate(axis=-1)([merge, bet_cnn]) 
    #print('bet_cnn.get_shape()', bet_cnn.get_shape()) # (None, 10000, 64)

    pooled = AttentionPooling1D(10)(bet_cnn) 
    #print('pooled.get_shape()', pooled.get_shape()) # (None, 1000, 64)

    reshaped = layers.Reshape((maxlen,1))(input7)
    mask_pooling = layers.MaxPool1D(10)(reshaped)
    mask_pooling = layers.Reshape((1000,))(mask_pooling)
    bet_cnn_mask = tf.keras.layers.Reshape((-1,))(mask_pooling)
    #print('bet_cnn_mask.get_shape()', bet_cnn_mask.get_shape())
    mut_mask = create_padding_mask(input6)
    bet_mask = create_padding_mask(bet_cnn_mask)

    mut = embedding_layer_mut([input6,input_mut])
    bet = embedding_layer_bet([bet_cnn_mask,pooled])
    #print('embedding_layer_mut.get_shape()', mut.get_shape()) # (None, 51, 128)
    #print('embedding_layer_bet.get_shape()', bet.get_shape()) # (None, 1000, 128)

    mut = trans_block_mut1(mut, mut_mask)
    bet = trans_block_bet1(bet, bet_mask)
    mut = trans_block_mut2(mut, mut_mask)
    bet = trans_block_bet2(bet, bet_mask)

    mut_bew = tf.keras.layers.Concatenate(axis=1)([mut, bet])
    mut_bew = tf.keras.layers.Conv1D(4, 1, kernel_initializer='he_uniform')(mut_bew)
    mut_bew = tf.keras.layers.GlobalAveragePooling1D()(mut_bew)
    output = tf.keras.layers.Dense(2, activation = 'softmax', name = 'output_softmax')(mut_bew)

    model = Model(inputs=[input1, input2, input3, input4, input5, input6, input7], outputs=output)
    return model



def build_EFM_large():
    # hyper-paramaters
    maxlen = 100000
    window_len = 51
    variant_box_len = 14
    block_size = 20
    vocab_size = 5
    embed_dim = 160
    num_heads = 8
    ff_dim = 256
    pos_embed_dim = 128+32
    seq_embed_dim_mut = 55+64+32
    seq_embed_dim_bet = 32

    ####### inputs (from data generator)
    input1 = Input(shape=(window_len, 4), name = 'input_before_51') # 51 seq-before-mutation
    input2 = Input(shape=(window_len, 4), name = 'input_after_51') # 51 seq-after-mutation
    input3 = Input(shape=(window_len,), name = 'input_atac_51') # 51 atac

    input4 = Input(shape=(maxlen, 4), name = 'input_bet_seq') # maxlen variant-tss-between-seq
    input5 = Input(shape=(maxlen,), name = 'input_atac_bet') # maxlen variant-tss-between-atac

    input6 = Input(shape=(window_len,), name = 'input_mask1') # mask-seq-51
    input7 = Input(shape=(maxlen,), name = 'input_mask2') # mask-seq-between

    ####### merge inputs of the same scale
    new_input3 = layers.Reshape((window_len,1), name = 'reshape_input_51')(input3)
    input_mut = layers.concatenate([input1, input2, new_input3], axis=-1)
    #print('input_mut.get_shape()', input_mut.get_shape()) # (None, 51, 9)
    new_input5 = layers.Reshape((maxlen,1), name = 'reshape_input_between')(input5)
    input_between = layers.concatenate([input4, new_input5], axis=-1)
    #print('input_between.get_shape()', input_between.get_shape()) # (None, maxlen, 5)

    # -----init all used basic layers-----
    leaky_relu = tf.keras.layers.LeakyReLU()
    embedding_layer_mut = TokenAndPositionEmbedding(window_len, vocab_size, embed_dim, pos_embed_dim, seq_embed_dim_mut)
    embedding_layer_bet = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim, pos_embed_dim, seq_embed_dim_bet)
    trans_block_mut1 = TransformerBlock(embed_dim, num_heads, ff_dim)
    trans_block_bet1 = TransformerBlock(embed_dim, num_heads, ff_dim)
    trans_block_mut2 = TransformerBlock(embed_dim, num_heads, ff_dim)
    trans_block_bet2 = TransformerBlock(embed_dim, num_heads, ff_dim)
    
    # 4 conv-resnet layers
    bet_cnn = tf.keras.layers.Conv1D(5, 16, kernel_initializer='he_uniform', padding='same')(input_between)
    bet_cnn = tf.keras.layers.Conv1D(16, 16, kernel_initializer='he_uniform', padding='same')(bet_cnn)
    merge = tf.keras.layers.Concatenate(axis=-1)([input_between, bet_cnn])
    bet_cnn = tf.keras.layers.Conv1D(64, 32, kernel_initializer='he_uniform', padding='same')(merge)
    bet_cnn = tf.keras.layers.Conv1D(128-21, 128, kernel_initializer='he_uniform', padding='same')(bet_cnn)
    bet_cnn = tf.keras.layers.Concatenate(axis=-1)([merge, bet_cnn])
    #print('bet_cnn.get_shape()', bet_cnn.get_shape()) # (None, 10000, 64)

    pooled = AttentionPooling1D(100)(bet_cnn) 
    #print('pooled.get_shape()', pooled.get_shape()) # (None, 1000, 64)

    reshaped = layers.Reshape((maxlen,1))(input7)
    mask_pooling = layers.MaxPool1D(100)(reshaped)
    mask_pooling = layers.Reshape((1000,))(mask_pooling)
    bet_cnn_mask = tf.keras.layers.Reshape((-1,))(mask_pooling)
    #print('bet_cnn_mask.get_shape()', bet_cnn_mask.get_shape())
    mut_mask = create_padding_mask(input6)
    bet_mask = create_padding_mask(bet_cnn_mask)

    mut = embedding_layer_mut([input6,input_mut])
    bet = embedding_layer_bet([bet_cnn_mask,pooled])
    #print('embedding_layer_mut.get_shape()', mut.get_shape()) # (None, 51, 128)
    #print('embedding_layer_bet.get_shape()', bet.get_shape()) # (None, 1000, 128)

    mut = trans_block_mut1(mut, mut_mask)
    bet = trans_block_bet1(bet, bet_mask)
    mut = trans_block_mut2(mut, mut_mask)
    bet = trans_block_bet2(bet, bet_mask)

    mut_bew = tf.keras.layers.Concatenate(axis=1)([mut, bet])
    mut_bew = tf.keras.layers.Conv1D(4, 1, kernel_initializer='he_uniform')(mut_bew)
    mut_bew = tf.keras.layers.GlobalAveragePooling1D()(mut_bew)
    output = tf.keras.layers.Dense(2, activation = 'softmax', name = 'output_softmax')(mut_bew)

    model = Model(inputs=[input1, input2, input3, input4, input5, input6, input7], outputs=output)
    return model