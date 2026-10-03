# Test `tf.pad` on an input whose rank is not resolved statically: `paddings` must hold one
# `[before, after]` row per input axis, so the result's rank is the number of rows even when the
# input's is unknown. An attention-image summary pads a weight tensor this way before splitting
# its last axis.
import tensorflow as tf


def consume_padded(x):
    assert x.shape == (1, 1, 1, 8) and x.dtype == tf.float32
    return x


def consume_padded_constant(x):
    assert x.shape == (3, 3, 1, 4) and x.dtype == tf.float32
    return x


def consume_padded_mixed(x):
    assert x.shape in [(1, 1, 1, 6), (2, 3, 4, 7)] and x.dtype == tf.float32
    return x


def pad_mixed(attn):
    consume_padded_mixed(tf.pad(attn, [[0, 0], [0, 0], [0, 0], [1, 1]]))


def consume_padded_union(x):
    assert x.shape in [(1, 1, 1, 6), (2, 3, 4, 7)] and x.dtype == tf.float32
    return x


def pad_union(use_unresolved):
    if use_unresolved:
        attn = tf.constant([[[values]]], dtype=tf.float32)
    else:
        attn = tf.ones((2, 3, 4, 5))
    consume_padded_union(tf.pad(attn, [[0, 0], [0, 0], [0, 0], [1, 1]]))


def pad_heads(attn):
    width = tf.math.mod(-tf.shape(attn)[1], 3)
    consume_padded(tf.pad(attn, [[0, 0], [0, 0], [0, 0], [0, width + 2]]))
    consume_padded_constant(tf.pad(attn, [[1, 1], [1, 1], [0, 0], [0, 0]]))


values = list(map(float, "1234"))
pad_heads(tf.constant([[[values]]], dtype=tf.float32))
pad_mixed(tf.constant([[[values]]], dtype=tf.float32))
pad_mixed(tf.ones((2, 3, 4, 5)))
pad_union(True)
pad_union(False)
