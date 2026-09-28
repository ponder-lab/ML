# Test for a loop-carried slice: the receiver of `context[1:]` is a phi of the slice's own result,
# so classifying the slice's origin through its receiver must not recurse forever.
import numpy as np
import tensorflow as tf


def consume(x):
    pass


def walk(levels, k, context, token):
    while k > 0:
        level = levels[k].get(context)
        if level is not None and token in level:
            return level[token]
        k -= 1
        context = context[1:]
    return levels[0][()][token]


levels = [{(): {1: 0.5}}, {(3,): {1: 0.25}}, {(2, 3): {1: 0.125}}]
for ctx, succ in list(levels[2].items()):
    shorter = ctx[1:]
    p = walk(levels, 1, shorter, 1)
assert p == 0.25

t = tf.constant(p)
assert t.shape == ()
assert t.dtype == tf.float32
consume(t)


def consume_np(x):
    pass


def shrink(v):
    while v.shape[0] > 2:
        v = v[1:]
    return v


a = shrink(np.ones((6, 2)))
assert isinstance(a, np.ndarray) and a.shape == (2, 2)
consume_np(a)
