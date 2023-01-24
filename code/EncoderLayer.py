import tensorflow as tf
from MultiHeadedAttention import *

class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self,  d_model, num_heads, dff, dropout):
    super(EncoderLayer, self).__init__()

    self.d_model = d_model
    self.num_heads = num_heads
    self.dff = dff
    self.dropout = dropout
    self.multi_head_attention =  MyMultiHeadAttention(d_model, num_heads)
    self.dropout_attention = tf.keras.layers.Dropout(dropout)
    self.add_attention = tf.keras.layers.Add()
    self.layer_norm_attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    self.dense1 = tf.keras.layers.Dense(dff, activation='relu')
    self.dense2 = tf.keras.layers.Dense(d_model)
    self.dropout_dense = tf.keras.layers.Dropout(dropout)
    self.add_dense = tf.keras.layers.Add()
    self.layer_norm_dense = tf.keras.layers.LayerNormalization(epsilon=1e-6)

  def call(self, inputs, mask=None, training=None):
    attention = self.multi_head_attention([inputs,inputs,inputs], mask = [mask,mask])
    attention = self.dropout_attention(attention, training = training)
    x = self.add_attention([inputs , attention])
    x = self.layer_norm_attention(x)

    ## Feed Forward
    dense = self.dense1(x)
    dense = self.dense2(dense)
    dense = self.dropout_dense(dense, training = training)
    x = self.add_dense([x , dense])
    x = self.layer_norm_dense(x)
    return x

  def get_config(self):
    
    config = super().get_config().copy()
    config.update({
      'd_model':self.d_model,
      'num_heads':self.num_heads,
      'dff':self.dff,
      'dropout':self.dropout
    })
  
    return config


  @classmethod
  def from_config(cls, config):
    return cls(**config)

          
