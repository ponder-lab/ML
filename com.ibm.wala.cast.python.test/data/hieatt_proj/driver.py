# Bespoke minimal driver over the VENDORED `BiLSTM` and `HieAttention` sources, reduced from the
# subject that exhibits the absent shape. The layer sources are copied verbatim; only the driver
# and the packaging are bespoke, so a difference in outcome isolates the setting from the source.
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
tokens = tf.constant(np.ones((batch_size, maxlen), dtype=np.int32))
out = model(tokens)
assert out.shape == (batch_size, 2 * hidden_dim), out.shape
