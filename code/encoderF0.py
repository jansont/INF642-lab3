import tensorflow as tf
from utils import *
from EncoderLayer import *
from tensorflow.keras import layers


class F0Encoder(tf.keras.layers.Layer):
   def __init__(self, input_vocab_size, num_layers, d_model, num_heads, dff, maximum_position_encoding = 10000, dropout = 0.0):     
    super(F0Encoder, self).__init__()
    self.input_vocab_size = input_vocab_size
    self.d_model = d_model
    self.num_layers = num_layers
    self.num_heads = num_heads
    self.dff = dff
    self.dropout=dropout
    self.maximum_position_encoding = maximum_position_encoding                
    self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model, mask_zero=True)
    self.conv1 = layers.Conv1D(d_model, 3, padding="same", activation="relu")
    self.conv2 = layers.Conv1D(d_model, 3, padding="same", activation="relu")
    self.conv3 = layers.Conv1D(d_model, 3, padding="same", activation="relu")                 
    self.dense = layers.Dense(d_model)
    self.pos = positional_encoding(maximum_position_encoding, d_model)
    self.encoder_layers = [EncoderLayer(d_model = d_model, num_heads = num_heads, dff = dff, dropout = dropout) for _ in range(num_layers)]
    self.dropout_ = tf.keras.layers.Dropout(dropout)

   def call(self, F0, mask=None, training=None):
    x = self.embedding(F0)
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.dropout_(x, training=training)

    # positional encoding
    x *= (tf.math.sqrt(tf.cast(self.d_model, tf.float32)))
    x += (self.pos[: , :tf.shape(x)[1], :])

    x = (self.dropout_(x, training=training))
    
    #Encoder layer
    embedding_mask = self.embedding.compute_mask(F0)
    for encoder_layer in self.encoder_layers:
      x = encoder_layer(x, mask = embedding_mask)    
    return x

   def compute_mask(self, inputs, mask=None):
    return self.embedding.compute_mask(inputs)


   def get_config(self):
      config = super().get_config().copy()
      config.update({
         'input_vocab_size':self.input_vocab_size,
         'num_layers':self.num_layers,
         'd_model':self.d_model,
         'num_heads':self.num_heads,
         'dff':self.dff,
         'maximum_position_encoding':self.maximum_position_encoding,
         'dropout':self.dropout
      })
            
      return config


   @classmethod
   def from_config(cls, config):
      return cls(**config)

           
