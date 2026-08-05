# Probe driver for wala/ML#618: the vendored `LayerNormalization` forward output in isolation.
# Analyzed statically, like `A.py` itself.
import tensorflow as tf

from layers.layer_norm import LayerNormalization


def consume(t):
    pass


ln = LayerNormalization(8)
out = ln(tf.ones((2, 3, 8)))
# Observed by running this file under `python3.10` (wala/ML#808), not read off the source.
assert out.shape == (2, 3, 8) and out.dtype == tf.float32
consume(out)
