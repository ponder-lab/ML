# Probe driver for wala/ML#618: the vendored `LayerNormalization` forward output in isolation.
# Analyzed statically, like `A.py` itself.
import tensorflow as tf

from layers.layer_norm import LayerNormalization


def consume(t):
    pass


ln = LayerNormalization(8)
out = ln(tf.ones((2, 3, 8)))
consume(out)
