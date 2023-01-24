import tensorflow as tf
from utils import *
from DecoderLayer import *
from tensorflow.keras import layers

class Decoder(tf.keras.layers.Layer):
  def __init__(self, target_vocab_size, num_layers, d_model, num_heads, dff, maximum_position_encoding, dropout):
    super(Decoder, self).__init__()
    self.target_vocab_size = target_vocab_size
    self.num_layers = num_layers
    self.num_heads = num_heads
    self.dff = dff
    self.maximum_position_encoding = maximum_position_encoding
    self.d_model = d_model
    self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model, mask_zero=True)
    self.pos = positional_encoding(maximum_position_encoding, d_model)
    self.decoder_layers = [ DecoderLayer(d_model = d_model, num_heads = num_heads, dff = dff, dropout = dropout)  for _ in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(dropout)

    
  def call(self, inputs, mask=None, training=None):
    x = self.embedding(inputs[0])
    
    # positional encoding
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x += self.pos[: , :tf.shape(x)[1], :]

    #Decoder layer
    embedding_mask = self.embedding.compute_mask(inputs[0])
    for decoder_layer in self.decoder_layers:
      x = decoder_layer([x,inputs[1]], mask = [embedding_mask, mask])    
    return x

  
  # Comment this out if you want to use the masked_loss()
  def compute_mask(self, inputs, mask=None):
    return self.embedding.compute_mask(inputs[0])

  def get_config(self):
    config = super().get_config().copy()
    config.update({
      'target_vocab_size':self.target_vocab_size,
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
