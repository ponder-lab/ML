# A matmul's `transpose_a`, `transpose_b`, `adjoint_a` and `adjoint_b` swap the last two axes of
# the operand they name before the product (wala/ML#1152).
import json

import numpy as np
import tensorflow as tf


def consume_transpose_b(t):
    return t


def consume_transpose_a(t):
    return t


def consume_adjoint_a(t):
    return t


def consume_adjoint_b(t):
    return t


def consume_positional(t):
    return t


def consume_false(t):
    return t


def consume_computed_flag(t):
    return t


def consume_unpacked(t):
    return t


q = tf.ones([2, 3, 4, 5])
k = tf.ones([2, 3, 6, 5])
transposed_b = tf.linalg.matmul(q, k, transpose_b=True)
consume_transpose_b(transposed_b)
assert transposed_b.shape == (2, 3, 4, 6)

transposed_a = tf.matmul(tf.ones([5, 4]), tf.ones([5, 6]), transpose_a=True)
consume_transpose_a(transposed_a)
assert transposed_a.shape == (4, 6)

adjoint_a = tf.matmul(tf.ones([5, 4]), tf.ones([5, 6]), adjoint_a=True)
consume_adjoint_a(adjoint_a)
assert adjoint_a.shape == (4, 6)

adjoint_b = tf.linalg.matmul(tf.ones([4, 5]), tf.ones([6, 5]), adjoint_b=True)
consume_adjoint_b(adjoint_b)
assert adjoint_b.shape == (4, 6)

# `transpose_a` and `transpose_b` passed positionally.
positional = tf.matmul(tf.ones([5, 4]), tf.ones([6, 5]), True, True)
consume_positional(positional)
assert positional.shape == (4, 6)

plain = tf.matmul(tf.ones([4, 5]), tf.ones([5, 6]), transpose_b=False)
consume_false(plain)
assert plain.shape == (4, 6)

# A flag whose value isn't a literal: of its two readings, only the transposed one contracts.
xs = [1]
flag = len(xs) > 0
computed = tf.matmul(tf.ones([4, 5]), tf.ones([6, 5]), transpose_b=flag)
consume_computed_flag(computed)
assert computed.shape == (4, 6)

# An operand of a statically unknown row count, bound by tuple-unpacking.
rows = np.array(json.loads("[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]"), np.float32).reshape(
    [-1, 5]
)
other, h = 0.0, rows
fed = tf.matmul(h, tf.ones([6, 5]), transpose_b=True)
consume_unpacked(fed)
assert fed.shape == (2, 6)
