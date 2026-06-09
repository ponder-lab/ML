import tensorflow as tf
import numpy as np

from deep_recommenders.keras.models.nlp.transformer import Transformer

# Driver for `Transformer.call` from
# `LongmaoTeamTf/deep_recommenders/keras/models/nlp/transformer.py`, a real-world
# sequence-to-sequence utility (a multi-head-attention encoder/decoder
# transformer), for tensor-type inference coverage. The `Transformer`,
# `MultiHeadAttention`, and supporting layers under `deep_recommenders/` are
# vendored verbatim from upstream; only this driver is bespoke. Exercises the
# multi-module import path the analyzer must follow.
vocab_size = 20
model_dim = 8
seq_len = 5
batch_size = 2

transformer = Transformer(
    vocab_size,
    model_dim,
    n_heads=2,
    encoder_stack=1,
    decoder_stack=1,
    feed_forward_size=16,
)

encoder_inputs = tf.constant(np.ones((batch_size, seq_len), dtype=np.int32))
decoder_inputs = tf.constant(np.ones((batch_size, seq_len), dtype=np.int32))
result = transformer(encoder_inputs, decoder_inputs)
assert (
    encoder_inputs.shape == (batch_size, seq_len) and encoder_inputs.dtype == tf.int32
)
assert (
    decoder_inputs.shape == (batch_size, seq_len) and decoder_inputs.dtype == tf.int32
)
assert result.shape == (batch_size, seq_len, vocab_size) and result.dtype == tf.float32
