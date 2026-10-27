import numpy as np
import tensorflow as tf


def consume_array(x):
    return x


def consume_cast(x):
    return x


class Graph:

    def __init__(self, x):
        self.x = Graph.cast_x(x)

    @classmethod
    def cast_x(cls, x):
        if isinstance(x, list):
            x = consume_array(np.array(x))
        if tf.is_tensor(x) and x.dtype == tf.float64:
            x = tf.cast(x, tf.float32)
        return x


def double(graph):
    return Graph(graph.x * 2.0)


class Observations:

    def __init__(self, stimulus_set):
        self.stimulus_set = consume_cast(stimulus_set.astype(np.int32))


def stack(first, second):
    return Observations(
        np.concatenate([first.stimulus_set, second.stimulus_set], axis=0)
    )


g = double(double(Graph([[0.0, 1.0, 2.0], [1.0, 2.0, 0.0]])))
assert g.x.shape == (2, 3)
assert g.x.dtype == np.float64

o = Observations(np.zeros((2, 3)))
o = stack(stack(o, o), o)
assert o.stimulus_set.shape == (6, 3)
assert o.stimulus_set.dtype == np.int32


def consume_grown(x):
    return x


def grow(a, n):
    x = a
    for i in range(n):
        if i > 0:
            x = consume_grown(np.concatenate([x, a], axis=0).astype(np.int32))
    return x


grown = grow(np.zeros((2, 3)), 3)
assert grown.shape == (6, 3)
assert grown.dtype == np.int32


def consume_recast(x):
    return x


def recast(a, n):
    x = a
    for i in range(n):
        if i > 0:
            x = consume_recast(x.astype(np.int32))
    return x


recast_ = recast(np.zeros((2, 3)), 3)
assert recast_.shape == (2, 3)
assert recast_.dtype == np.int32


def consume_zipped(x):
    return x


def zip_loop(n):
    ds = tf.data.Dataset.from_tensor_slices(tf.ones([4, 3]))
    for i in range(n):
        if i > 0:
            ds = tf.data.Dataset.zip((ds, ds)).map(lambda a, b: a + b)
    for element in ds:
        assert element.shape == (3,)
        assert element.dtype == tf.float32
        consume_zipped(element)


zip_loop(3)
