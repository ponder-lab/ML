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


# A two-dot relative import from a subpackage, `from ..conv.dense import Layer`.
from pkg.layers.sub.make import make


def consume_two_dot(t):
    assert t.shape == (2, 3) and t.dtype == tf.float32, (t.shape, t.dtype)


consume_two_dot(make()(tf.ones((2, 3))))

# A sibling module imported from its own package, `from . import dense`, beside a same-named one.
from pkg.nn.conv.use import use


def consume_sibling(t):
    assert t.shape == (5,) and t.dtype == tf.float32, (t.shape, t.dtype)


consume_sibling(use())

# A three-level relative import, `from .a.b.c import deep`, beside a same-named `pkg/other/a/b/c.py`.
from pkg.deep import deep
from pkg.other import deep as other_deep


def consume_three(t):
    assert t.shape == (7,) and t.dtype == tf.float32, (t.shape, t.dtype)


def consume_other_three(t):
    assert t.shape == (9, 9) and t.dtype == tf.float32, (t.shape, t.dtype)


consume_three(deep())
consume_other_three(other_deep())
