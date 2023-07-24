# model structure 

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

from .utils.attention_RoPE import MultiHeadedAttentionLayer
from .utils.rotary_embedding import apply_rotary_emb, RotaryEmbedding
from .utils import utils_bigbird as bigbird_utils
from .utils import attention_RoPE as bigbird_attention

class TransformerBlock(layers.Layer):
    def __init__(self, seq_length, block_size, size_per_head,  
                 embed_dim, num_heads, ff_dim, initializer_range=0.02,rate=0.1,use_bias=True):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadedAttentionLayer(attention_type='block_sparse',
                                             #attention_type='original_full',
                                             num_attention_heads=num_heads,
                                             size_per_head=size_per_head,
                                             num_rand_blocks=3,
                                             from_seq_length=seq_length,
                                             to_seq_length=seq_length,
                                             from_block_size=block_size,
                                             to_block_size=block_size,                                             
                                             name="self")
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(size_per_head*num_heads),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        self.projection_layer = bigbird_utils.Dense3dProjLayer(
                      num_heads, size_per_head,
                      bigbird_utils.create_initializer(initializer_range), None,
                      "dense", use_bias)   

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim

    def get_config(self):
        config = super().get_config().copy()
        config.update({
              'embed_dim': self.embed_dim,
              'num_heads': self.num_heads,
              'ff_dim': self.ff_dim,
            })
        return config   

    def call(self, inputs, attention_mask=None, band_mask=None, from_mask=None, to_mask=None, input_blocked_mask=None, training=True): # transformer encoder
        # masks:[attention_mask, band_mask, from_mask, to_mask, input_blocked_mask]
        attn_output = self.att(inputs, inputs, [
            attention_mask, band_mask, from_mask, to_mask, input_blocked_mask, input_blocked_mask], training=training)
        attn_output = self.dropout1(attn_output, training=training)
        attn_output = self.projection_layer(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    #print(seq.shape)
    # add extra dimensions to add the padding
    # to the attention logits.
    return  seq[:, tf.newaxis, tf.newaxis, :]# (batch_size, 1, 1, seq_len)

def create_bbmask(ori_mask, maxlen, block_size):
    # attention_mask:[batch_size,seq_length, seq_length]
    # from_mask:[batch_size, 1, seq_length, 1]
    # to_mask:[batch_size, 1, 1, seq_length]
    q_mask = tf.expand_dims(ori_mask, axis=-1)  # [seq_len, 1]
    k_mask = layers.Reshape((1, -1))(q_mask) # [1, seq_len] 
    attention_mask = tf.matmul(q_mask, k_mask)   # [seq_len, seq_len] 
    from_mask = ori_mask[:, tf.newaxis, :, tf.newaxis]
    #print(from_mask.get_shape())
    to_mask = ori_mask[:, tf.newaxis, tf.newaxis, :]
    block_mask = layers.Reshape((maxlen//block_size, block_size))(from_mask)
    band_mask = bigbird_attention.create_band_mask_from_inputs(block_mask,block_mask)
    return [attention_mask, from_mask, to_mask, block_mask, band_mask]

def build_EMO_middle():

    # hyper-paramaters
    maxlen = 10000
    window_len = 51
    block_size = 20
    embed_dim = 64
    num_heads = 4
    ff_dim = 256

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

    ####### between branch
    pos_emb_bet = RotaryEmbedding(dim = 32)
    freqs = pos_emb_bet(tf.range(maxlen), cache_key = maxlen)
    ini_position = tf.random.normal((1, maxlen, 64)) # queries - (batch, seq len, dimension of head)
    freqs = freqs[None, ...] # expand dimension for batch dimension
    ini_position = apply_rotary_emb(freqs, ini_position)
    ini_position = tf.reshape(ini_position,[maxlen, 64])
    position_embedding_between_seq = tf.keras.layers.Embedding(input_dim=maxlen, output_dim=64, trainable=False,
                                        weights=[ini_position])(input_between)
    #print('position_embedding_between_seq.get_shape()', position_embedding_between_seq.get_shape()) # (None, maxlen, 5, 64)
    position_embedding_between_seq = tf.keras.layers.Reshape((maxlen, -1), name = 'reshape_embedding_between_seq')(position_embedding_between_seq)
    #print('position_embedding_between_seq.get_shape()', position_embedding_between_seq.get_shape()) # (None, maxlen, 320)
    
    
    fc = layers.Dense(128)(position_embedding_between_seq)
    #print('gru_out.get_shape()', gru_out.get_shape()) # (None, maxlen, 128)

    # average pooling for between profile
    pooled = layers.AveragePooling1D(10)(fc) 
    #print('pooled.get_shape()', pooled.get_shape()) # (None, 1000, 128)

    # max pooling for between masks
    reshaped = layers.Reshape((maxlen,1))(input7)
    mask_pooling = layers.MaxPool1D(10)(reshaped)
    mask_pooling = layers.Reshape((1000,))(mask_pooling)
    #print('mask_pooling.get_shape()', mask_pooling.get_shape()) # (None, 1000)

    [attention_mask_bet, from_mask_bet, to_mask_bet, block_mask_bet, band_mask_bet] = create_bbmask(mask_pooling, 1000, block_size)
    trans_block_bet1 = TransformerBlock(1000, block_size, pooled.get_shape()[-1]//num_heads, embed_dim, num_heads, ff_dim)
    bet1 = trans_block_bet1(pooled, attention_mask=attention_mask_bet, band_mask=band_mask_bet, from_mask=from_mask_bet, to_mask=to_mask_bet, input_blocked_mask=block_mask_bet)
    #print('bet1.get_shape()', bet1.get_shape()) # bet1.get_shape() (None, 1000, 128)

    trans_block_bet2 = TransformerBlock(1000, block_size, bet1.get_shape()[-1]//num_heads, embed_dim, num_heads, ff_dim)
    bet2 = trans_block_bet2(bet1, attention_mask=attention_mask_bet, band_mask=band_mask_bet, from_mask=from_mask_bet, to_mask=to_mask_bet, input_blocked_mask=block_mask_bet)
    bet3 = trans_block_bet2(bet2, attention_mask=attention_mask_bet, band_mask=band_mask_bet, from_mask=from_mask_bet, to_mask=to_mask_bet, input_blocked_mask=block_mask_bet)
    #print('bet3.get_shape()', bet3.get_shape()) # bet3.get_shape() (None, 1000, 128)

    ###### mutation branch
    conv_mut = layers.Conv1D(32, 3, kernel_initializer='he_uniform')(input_mut) # (None, 49, 32)
    conv_mut = layers.Conv1D(16, 3, kernel_initializer='he_uniform')(conv_mut) # (None, 47, 16)
    dense_mut = layers.Flatten()(conv_mut)
    dense_mut = layers.Dense(128)(dense_mut)
    dense_mut = layers.Reshape((1, 128))(dense_mut) # (None, 1, 128)
    #print('dense_mut.get_shape()', dense_mut.get_shape()) 

    ##### merge between & mutation branch
    merged = layers.concatenate([bet3, dense_mut], axis=1)
    #print('merged.get_shape()', merged.get_shape()) # (None, 1001, 128)
    merged = layers.Dense(128)(merged)
    merged = layers.Dense(32)(merged)
    merged = layers.Flatten()(merged)
    merged = layers.Dense(32)(merged)
    output = layers.Dense(2, activation = 'softmax')(merged)
    
    model = Model(inputs=[input1, input2, input3, input4, input5, input6, input7], outputs=output)
    return model