# Driver for `DynamicPositionEmbedding.call` from
# `jason9693/MusicTransformer-tensorflow2.0/custom/layers.py` (the wala/ML#676 subject): `inputs`
# receives the embedded-and-scaled token tensor from `Encoder.call`'s `x = self.pos_encoding(x)`,
# where `x` is loop-carried through the encoder layers below it. The module is vendored verbatim
# from upstream; only this driver is bespoke.
import numpy as np
import tensorflow as tf

from custom.layers import Encoder

num_layers = 2
d_model = 64
vocab_size = 100
max_len = 50
batch = 2

enc = Encoder(
    num_layers=num_layers,
    d_model=d_model,
    input_vocab_size=vocab_size,
    max_len=max_len,
)
tokens = tf.constant(np.ones((batch, max_len), dtype=np.int32))
out, weights = enc(tokens, mask=None, training=False)

assert out.shape == (batch, max_len, d_model)
assert out.dtype == tf.float32
assert len(weights) == num_layers
