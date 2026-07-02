# Probe driver for wala/ML#618: the vendored `Gpt2` forward output in isolation (no training
# loop, no `GradientTape`, no `fit` indirection). Analyzed statically, like `A.py` itself
# (whose module-level driver needs a tfrecord setup to run).
import tensorflow as tf

from A import Gpt2


def consume(t):
    pass


model = Gpt2(
    num_layers=2, d_model=8, num_heads=2, dff=16, max_seq_len=12, vocab_size=10
)
inputs = tf.constant([[1, 2, 3], [4, 5, 6]])
logits, _ = model(inputs, training=False)
consume(logits)
assert logits.shape == (2, 3, 10)
assert logits.dtype == tf.float32
