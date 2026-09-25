# Probe driver for the sampling loop: the vendored `Gpt2` called with the `past` its previous call
# returned, so every decoder layer's `past` and the attention layer's `past_layer` are typed by a
# value fed back around the loop. Analyzed statically, like `A.py` itself.
import tensorflow as tf

from A import Gpt2


def consume(t):
    pass


model = Gpt2(
    num_layers=2, d_model=8, num_heads=2, dff=16, max_seq_len=12, vocab_size=10
)
prev = tf.constant([[1, 2, 3], [4, 5, 6]])
past = None
for i in range(3):
    logits, past = model(prev, training=False, past=past)
    consume(logits)
    prev = tf.constant([[7], [8]])
assert logits.shape == (2, 1, 10) and logits.dtype == tf.float32
assert (
    len(past) == 2 and past[0].shape == (2, 2, 2, 5, 4) and past[0].dtype == tf.float32
)
