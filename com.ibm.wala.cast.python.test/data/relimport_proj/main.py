# A package re-exports a class through a relative import, `from .conv.dense import Layer`, while a
# sibling package holds a module of the same name, `pkg/nn/conv/dense.py`. The relative import names
# the importer's own package's module, `pkg/layers/conv/dense.py`, wherever the project is checked out.
import tensorflow as tf

from pkg.layers import Layer
from pkg.nn import dense


def consume_scaled(t):
    assert t.shape == (2, 3) and t.dtype == tf.float32, (t.shape, t.dtype)


def consume_dense(t):
    assert t.shape == (5,) and t.dtype == tf.float32, (t.shape, t.dtype)


layer = Layer()
consume_scaled(layer(tf.ones((2, 3))))
consume_dense(dense())
