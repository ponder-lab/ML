# Test for https://github.com/wala/ML/issues/1297: a `tf.nest` call whose result flows back into
# its own structure argument. The analysis copies the structure at the call, and the copy reaches
# the call again, so the copies must still be finite.
import collections

import tensorflow as tf

Pair = collections.namedtuple("Pair", ["left", "right"])


def shape_of(t):
    return tuple(t.shape)


# Each call below maps a function of its own: a function shared between calls is analyzed once for
# all of them, so each call's result would read every call's.


def step_looped(t):
    return t + 1


def step_first(t):
    return t + 1


def step_second(t):
    return t * 2


def step_named(t):
    return t + 1


def pair_up(t):
    return (t, t)


def step_once(t):
    return t + 1


def sum_rows(t):
    return tf.reduce_sum(t, axis=1)


def of_looped(t):
    assert shape_of(t) == (2, 3)
    return t


def of_looped_inner(t):
    assert shape_of(t) == (4,)
    return t


def of_repacked(t):
    assert shape_of(t) == (3, 6)
    return t


def of_alternated(t):
    assert shape_of(t) == (2, 8)
    return t


def of_named(t):
    assert shape_of(t) == (5, 7)
    return t


def of_deepened(t):
    assert shape_of(t) == (2, 9)
    return t


def of_once(t):
    assert shape_of(t) == (6, 2)
    return t


def of_twice(t):
    assert shape_of(t) == (6,)
    return t


# One call in a loop, over a tuple that holds a list.
looped = (tf.ones([2, 3]), [tf.zeros([4]), tf.ones([5])])
for _ in range(3):
    looped = tf.nest.map_structure(step_looped, looped)
of_looped(looped[0])
of_looped_inner(looped[1][0])

# A structure packed around its own leaves in a loop.
repacked = (tf.ones([3, 6]), [tf.zeros([7])])
for _ in range(3):
    repacked = tf.nest.pack_sequence_as(repacked, tf.nest.flatten(repacked))
of_repacked(repacked[0])

# Two calls in a loop, each given the other's result.
alternated = {"a": tf.ones([2, 8]), "b": (tf.zeros([9]),)}
for _ in range(2):
    alternated = tf.nest.map_structure(step_first, alternated)
    alternated = tf.nest.map_structure(step_second, alternated)
of_alternated(alternated["a"])

# A namedtuple instance, read by a field's name after the loop.
named = Pair(left=tf.ones([5, 7]), right=tf.zeros([3, 8]))
for _ in range(2):
    named = tf.nest.map_structure(step_named, named)
of_named(named.left)

# A function that returns a structure, so every turn of the loop nests the result once more.
deepened = [tf.ones([2, 9])]
for _ in range(2):
    deepened = tf.nest.map_structure(pair_up, deepened)
of_deepened(deepened[0][0][0])

# Two calls in a row and no loop: the second call's copy holds the second function's results alone.
chained = (tf.ones([6, 2]), tf.zeros([6, 3]))
once = tf.nest.map_structure(step_once, chained)
twice = tf.nest.map_structure(sum_rows, once)
of_once(once[0])
of_twice(twice[0])


# The structures themselves: after its loop each variable holds one copy, made by the call whose
# result the loop leaves in it, and no copy of a copy.
def holds_looped(s):
    return s


def holds_repacked(s):
    return s


def holds_alternated(s):
    return s


def holds_named(s):
    return s


def holds_deepened(s):
    return s


holds_looped(looped)
holds_repacked(repacked)
holds_alternated(alternated)
holds_named(named)
holds_deepened(deepened)
