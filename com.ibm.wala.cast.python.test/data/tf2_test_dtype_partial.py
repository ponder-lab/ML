# Partial dtype unions (wala/ML#862). `operand` joins two allocations in one frame, so no calling
# context separates them; a dtype looked up by a name parsed at run time is one the analysis cannot
# read. With one known and one unreadable dtype, the joined value's dtype is known at one allocation
# and unknown at the other, and each operation below must keep the known member beside the unknown
# one rather than read as unknown alone.
import json

import tensorflow as tf


def runtime_dtype(name):
    return getattr(tf, json.loads(name))


def operand(flag, first, second, shape):
    if flag:
        return tf.ones(shape, dtype=first)
    return tf.ones(shape, dtype=second)


def consume_partial_operand(x):
    pass


def consume_partial_matmul(x):
    pass


def consume_partial_elementwise(x):
    pass


def consume_partial_cast(x):
    pass


def consume_partial_pass_through(x):
    pass


def consume_partial_reduce(x):
    pass


def partial_ops(flag):
    p = operand(flag, tf.float32, runtime_dtype('"float32"'), [2, 3])
    assert p.shape == (2, 3)
    assert p.dtype == tf.float32
    consume_partial_operand(p)

    r = p @ tf.ones([3, 4])
    assert r.shape == (2, 4)
    assert r.dtype == tf.float32
    consume_partial_matmul(r)

    q = p * 2.0
    assert q.shape == (2, 3)
    assert q.dtype == tf.float32
    consume_partial_elementwise(q)

    c = tf.cast(p, tf.int64)
    assert c.shape == (2, 3)
    assert c.dtype == tf.int64
    consume_partial_cast(c)

    u = tf.transpose(tf.identity(p))
    assert u.shape == (3, 2)
    assert u.dtype == tf.float32
    consume_partial_pass_through(u)

    idx = operand(flag, tf.int32, runtime_dtype('"int32"'), [5])
    n = tf.reduce_max(idx) + 1
    assert n.shape == ()
    assert n.dtype == tf.int32
    consume_partial_reduce(n)


def consume_known_pass_through(x):
    pass


def consume_known_reduce(x):
    pass


def known_ops(flag):
    p = operand(flag, tf.float32, tf.float32, [2, 3])
    u = tf.transpose(tf.identity(p))
    assert u.shape == (3, 2)
    assert u.dtype == tf.float32
    consume_known_pass_through(u)

    idx = operand(flag, tf.int32, tf.int32, [5])
    n = tf.reduce_max(idx) + 1
    assert n.shape == ()
    assert n.dtype == tf.int32
    consume_known_reduce(n)


def consume_unknown_pass_through(x):
    pass


def consume_unknown_reduce(x):
    pass


def unknown_ops(flag):
    p = operand(flag, runtime_dtype('"float32"'), runtime_dtype('"float32"'), [2, 3])
    u = tf.transpose(tf.identity(p))
    assert u.shape == (3, 2)
    assert u.dtype == tf.float32
    consume_unknown_pass_through(u)

    idx = operand(flag, runtime_dtype('"int32"'), runtime_dtype('"int32"'), [5])
    n = tf.reduce_max(idx) + 1
    assert n.shape == ()
    assert n.dtype == tf.int32
    consume_unknown_reduce(n)


flag = len(json.loads("[]")) == 0
partial_ops(flag)
known_ops(flag)
unknown_ops(flag)
