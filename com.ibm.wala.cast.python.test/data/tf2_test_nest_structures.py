# Test for https://github.com/wala/ML/issues/1067.
import collections

import tensorflow as tf
from tensorflow.python.util import nest

Pair = collections.namedtuple("Pair", ["left", "right"])


def shape_of(t):
    return tuple(t.shape)


def on_dict(t):
    assert shape_of(t) in {(1, 2), (3,)}
    return t


def on_tuple(t):
    assert shape_of(t) in {(2, 2), (4,)}
    return t


def on_list(t):
    assert shape_of(t) in {(5,), (6, 1)}
    return t


def on_nested(t):
    assert shape_of(t) in {(7,), (2, 3), (3, 4)}
    return t


def on_tensor(t):
    assert shape_of(t) == (2, 4)
    return t


def on_namedtuple(t):
    assert shape_of(t) in {(3, 2), (4, 3)}
    return t


def on_private_path(t):
    assert shape_of(t) == (3, 3)
    return t


def from_lambda(t):
    assert shape_of(t) == (4, 4)
    return t


def on_appended(t):
    assert shape_of(t) == (5, 5)
    return t


def on_two(a, b):
    assert shape_of(a) == (5, 6)
    assert shape_of(b) == (3, 6)
    return a


def sized(t):
    assert shape_of(t) == (3,)


def on_default(t, n=3):
    assert shape_of(t) == (6,)
    sized(tf.ones([n]))
    return t


def star_first(t):
    assert shape_of(t) == (2, 1)


def on_star(*args):
    assert len(args) == 1
    star_first(args[0])


def on_three(a, b, c):
    assert shape_of(a) == (7, 1)
    assert shape_of(b) == (7, 2)
    assert shape_of(c) == (7, 3)
    return a


def on_keyword(t):
    assert shape_of(t) == (8, 2)
    return t


def member_row(r):
    assert shape_of(r) == (3,)


def member_rows(r):
    assert shape_of(r) == (1, 3)


def on_member(t):
    assert shape_of(t) == (2, 3)
    member_row(t[0])
    member_rows(t[1:])
    return t


def reshaped(r):
    assert shape_of(r) == (6,)


def summed(r):
    assert shape_of(r) == (2, 3)


def expanded(r):
    assert shape_of(r) == (1, 2, 3)


def on_op(t):
    reshaped(tf.reshape(t, [6]))
    summed(t + 1.0)
    expanded(tf.expand_dims(t, 0))
    return t


def flat_subscript(t):
    assert shape_of(t) in {(3, 5), (4, 5)}


def flat_iteration(t):
    assert shape_of(t) in {(3, 5), (4, 5)}


tf.nest.map_structure(on_dict, {"a": tf.ones([1, 2]), "b": tf.zeros([3], tf.int32)})
tf.nest.map_structure(on_tuple, (tf.ones([2, 2]), tf.ones([4])))
tf.nest.map_structure(on_list, [tf.ones([5]), tf.ones([6, 1])])
tf.nest.map_structure(
    on_nested, {"a": (tf.ones([7]), [tf.ones([2, 3])]), "b": {"c": tf.ones([3, 4])}}
)
tf.nest.map_structure(on_tensor, tf.ones([2, 4]))
tf.nest.map_structure(on_namedtuple, Pair(tf.ones([3, 2]), tf.ones([4, 3])))
nest.map_structure(on_private_path, [tf.ones([3, 3])])
tf.nest.map_structure(lambda t: from_lambda(t), (tf.ones([4, 4]),))

appended = []
appended.append(tf.ones([5, 5]))
tf.nest.map_structure(on_appended, appended)

tf.nest.map_structure(on_two, [tf.ones([5, 6])], [tf.ones([3, 6])])

tf.nest.map_structure(on_default, [tf.ones([6])])
tf.nest.map_structure(on_star, [tf.ones([2, 1])])

tf.nest.map_structure(
    on_three, (tf.ones([7, 1]),), (tf.ones([7, 2]),), (tf.ones([7, 3]),)
)
tf.nest.map_structure(on_keyword, [tf.ones([8, 2])], expand_composites=True)

tf.nest.map_structure(on_member, [tf.ones([2, 3])])
tf.nest.map_structure(on_op, {"a": tf.ones([2, 3])})

flat = tf.nest.flatten({"a": tf.ones([3, 5]), "b": (tf.ones([4, 5]),)})
assert len(flat) == 2
flat_subscript(flat[0])
for leaf in flat:
    flat_iteration(leaf)
