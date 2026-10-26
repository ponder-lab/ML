# Probe driver for the sampling step's `logits[:, -1, :]` (wala/ML#1153): the subscript of the
# vendored `Gpt2`'s logits keeps only the shapes its receiver's members give. Analyzed statically,
# like `A.py` itself.
import tensorflow as tf

from A import Gpt2


def top_k_logits(logits, k):
    assert logits.shape == (1, 10)
    return logits


def consume_unpacked(t):
    return t


def consume_sliced(t):
    return t


def consume_divided(t):
    return t


def sample(model, temperature):
    prev = tf.constant([[1, 2, 3]])
    logits, past = model(prev, training=False, past=None)
    consume_unpacked(logits)
    consume_sliced(logits[:, -1, :])
    consume_divided(logits[:, -1, :] / temperature)
    logits = logits[:, -1, :] / temperature
    return top_k_logits(logits, 2)


model = Gpt2(
    num_layers=2, d_model=8, num_heads=2, dff=16, max_seq_len=12, vocab_size=10
)
sample(model, 1.0)
