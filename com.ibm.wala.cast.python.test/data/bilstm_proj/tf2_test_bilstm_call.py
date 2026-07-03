# Driver for `BiLSTM.call` from `kyzhouhzau/NLPGNN/nlpgnn/layers/bilstm.py` (the wala/ML#618
# row): `inputs` receives an integer token-ID tensor feeding a Keras `Embedding`. The `BiLSTM`
# layer is vendored verbatim from upstream; only this driver is bespoke.
import numpy as np
import tensorflow as tf

from nlpgnn.layers.bilstm import BiLSTM

maxlen = 5
vocab_size = 10
embedding_dims = 8
hidden_dim = 4
batch = 2

tokens = tf.constant(np.ones((batch, maxlen), dtype=np.int32))
layer = BiLSTM(maxlen, vocab_size, embedding_dims, hidden_dim)
out = layer(tokens, training=False)

assert tokens.shape == (batch, maxlen)
assert tokens.dtype == tf.int32
assert out.shape == (batch, maxlen, 2 * hidden_dim)
assert out.dtype == tf.float32
