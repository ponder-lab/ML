# Test for https://github.com/wala/ML/issues/1280: what `tf.nest.map_structure` and
# `tf.nest.pack_sequence_as` return.
import collections
import json

import tensorflow as tf

Pair = collections.namedtuple("Pair", ["left", "right"])


def shape_of(t):
    return tuple(t.shape)


# Each call below maps a function of its own: a function shared between calls is analyzed once for
# all of them, so each call's result would read every call's.


def double(t):
    return t * 2


def of_dict(t):
    assert shape_of(t) == (1, 2)
    return t


def of_tuple(t):
    assert shape_of(t) == (2, 2)
    return t


def of_list(t):
    assert shape_of(t) == (5,)
    return t


def of_appended(t):
    assert shape_of(t) == (2, 6)
    return t


def of_concatenated(t):
    assert shape_of(t) == (2, 8)
    return t


def of_repeated(t):
    assert shape_of(t) == (3, 8)
    return t


def of_iterated(t):
    assert shape_of(t) in {(5,), (6, 1)}
    return t


def of_nested(t):
    assert shape_of(t) == (2, 3)
    return t


def of_inner(inner):
    assert isinstance(inner, tuple)
    return inner


def of_same_inner(inner):
    assert isinstance(inner, list)
    return inner


def of_same_deep(t):
    assert shape_of(t) == (2, 9)
    return t


def of_namedtuple(t):
    assert shape_of(t) == (3, 2)
    return t


def of_twice(t):
    assert shape_of(t) == (4, 5)
    return t


def of_tensor(t):
    assert shape_of(t) == (2, 4)
    return t


def of_replaced(t):
    assert shape_of(t) == (7,)
    return t


def of_source(t):
    assert shape_of(t) == (1, 3)
    return t


def of_structure(d):
    assert isinstance(d, dict)
    return d


def of_two(t):
    assert shape_of(t) == (5, 6)
    return t


def of_first(t):
    assert shape_of(t) == (5, 7)
    return t


def of_unread(t):
    assert shape_of(t) == (6,)
    return t


def of_parameter(t):
    assert shape_of(t) == (4, 4)
    return t


def mapped(structure):
    return tf.nest.map_structure(double, structure)


def of_none(t):
    assert shape_of(t) == (8,)
    return t


def of_none_place(v):
    assert v is None
    return v


def keep(t):
    return t


def of_packed(t):
    assert shape_of(t) == (2,)
    return t


def of_packed_tensor(t):
    assert shape_of(t) == (9,)
    return t


def of_packed_structure(d):
    assert isinstance(d, dict)
    return d


def of_repacked(t):
    assert shape_of(t) == (2, 7)
    return t


def of_element(t):
    assert shape_of(t) == (3,)
    return t


def of_product(t):
    assert shape_of(t) == (2, 5)
    return t


of_dict(
    tf.nest.map_structure(lambda t: t * 2, {"a": tf.ones([1, 2]), "b": tf.ones([3])})[
        "a"
    ]
)
of_tuple(tf.nest.map_structure(lambda t: t * 2, (tf.ones([2, 2]), tf.ones([4])))[0])
doubled = tf.nest.map_structure(lambda t: t * 2, [tf.ones([5]), tf.ones([6, 1])])
of_list(doubled[0])
for member in doubled:
    of_iterated(member)
appended = []
appended.append(tf.ones([2, 6]))
of_appended(tf.nest.map_structure(lambda t: t * 2, appended)[0])
for member in tf.nest.map_structure(
    lambda t: t * 2, [tf.ones([2, 8])] + [tf.ones([2, 8])]
):
    of_concatenated(member)
for member in tf.nest.map_structure(lambda t: t * 2, [tf.ones([3, 8])] * 2):
    of_repeated(member)
deep = tf.nest.map_structure(
    lambda t: t * 2, {"a": (tf.ones([7]), tf.ones([2, 3])), "b": tf.ones([3, 4])}
)
of_nested(deep["a"][1])
of_inner(deep["a"])
# A list in a list: one call copies both, and they are copied apart though they share a class.
same = tf.nest.map_structure(lambda t: t * 2, [[tf.ones([2, 9])]])
of_same_inner(same[0])
of_same_deep(same[0][0])
of_namedtuple(
    tf.nest.map_structure(lambda t: t * 2, Pair(tf.ones([3, 2]), tf.ones([4, 3]))).left
)
# A namedtuple instance the call returned, mapped again, keeps its names.
once = tf.nest.map_structure(lambda t: t * 2, Pair(tf.ones([3, 5]), tf.ones([4, 5])))
of_twice(tf.nest.map_structure(lambda t: t * 2, once).right)
of_tensor(tf.nest.map_structure(lambda t: t * 2, tf.ones([2, 4])))

# The call returns the function's results, not the values the function was given, and leaves the
# structure it was given as it was.
source = {"a": tf.ones([1, 3])}
replaced = tf.nest.map_structure(lambda t: tf.zeros([7]), source)
of_replaced(replaced["a"])
of_source(source["a"])
of_structure(replaced)

of_two(
    tf.nest.map_structure(lambda a, b: a + b, (tf.ones([5, 6]),), (tf.ones([5, 6]),))[0]
)
# A control: a function of two structures whose result is its first leaf.
of_first(
    tf.nest.map_structure(lambda a, b: a, (tf.ones([5, 7]),), (tf.ones([5, 7]),))[0]
)
of_parameter(mapped({"a": tf.ones([4, 4])})["a"])

# A `None` is a leaf: the function is called on it, and its result takes the place.
gaps = tf.nest.map_structure(lambda t: tf.zeros([8]), {"a": None})
of_none(gaps["a"])
kept = tf.nest.map_structure(keep, {"a": None})
of_none_place(kept["a"])

# A structure the analysis reads nothing for.
of_unread(tf.nest.map_structure(lambda t: tf.zeros([6]), json.loads("[1.0]"))[0])

packed = tf.nest.pack_sequence_as({"a": 0, "b": 0}, [tf.ones([2]), tf.ones([3, 1])])
of_packed(packed["a"])
of_packed_structure(packed)
of_packed_tensor(tf.nest.pack_sequence_as(0, [tf.ones([9])]))
whole = {"a": tf.ones([2, 7])}
of_repacked(tf.nest.pack_sequence_as(whole, tf.nest.flatten(whole))["a"])

# The same over a dataset's element, in the function `map` calls.
rows = tf.data.Dataset.from_tensor_slices({"a": tf.ones([4, 3])})
for row in rows.map(lambda e: tf.nest.map_structure(lambda t: t * 2, e)):
    of_element(row["a"])

# A leaf that an operator computed.
of_product(tf.nest.map_structure(lambda t: t * 2, tf.ones([2, 5]) * tf.ones([2, 5])))
