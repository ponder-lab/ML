# The third arm: fed from a dataset whose element is a SINGLE tensor rather than a tuple, so no
# destructuring occurs. Separates "the value came from a dataset" from "the value came from a
# destructured tuple element".
import numpy as np
import tensorflow as tf

from bilstm import BiLSTM
from attention import HieAttention

maxlen = 128
batch_size = 64
embedding_dims = 100
hidden_dim = 50
vocab_size = 30522


class BilstmAttention(tf.keras.Model):
    def __init__(
        self,
        maxlen,
        vocab_size,
        embedding_dims,
        hidden_dim,
        dropout_rate=0.5,
        return_state=False,
        return_sequences=True,
        weights=None,
        **kwargs
    ):
        super(BilstmAttention, self).__init__(**kwargs)
        self.bilstm = BiLSTM(
            maxlen,
            vocab_size,
            embedding_dims,
            hidden_dim,
            dropout_rate=dropout_rate,
            return_state=return_state,
            return_sequences=return_sequences,
            weights=weights,
        )
        self.att = HieAttention(2 * hidden_dim, attention_size=100)

    def call(self, inputs, training=True):
        logits = self.bilstm(inputs, training)
        logits, _ = self.att(logits)
        return logits


model = BilstmAttention(maxlen, vocab_size, embedding_dims, hidden_dim)

rows = 128
ids = tf.constant(np.ones((rows, maxlen), dtype=np.int32))
token_type_ids = tf.constant(np.ones((rows, maxlen), dtype=np.int32))
input_masks = tf.constant(np.ones((rows, maxlen), dtype=np.int32))
labels = tf.constant(np.ones((rows,), dtype=np.int32))

loaded = tf.data.Dataset.from_tensor_slices(ids).batch(batch_size)

for X in loaded:
    out = model(X)
    assert out.shape == (batch_size, 2 * hidden_dim), out.shape
