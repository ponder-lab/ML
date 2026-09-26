import tensorflow as tf


def consume_int(a):
    pass


def consume_float(b):
    pass


def consume_tags(c):
    pass


def consume_mixed(d):
    pass


lengths = tf.constant([2, 3], dtype=tf.int32)

# Positional input with a keyword `seq_axis`: shape and dtype pass through -> (2, 3) int32.
x = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.int32)
a = tf.reverse_sequence(x, lengths, seq_axis=1)
assert a.shape == (2, 3) and a.dtype == tf.int32
consume_int(a)

# Keyword input with both axes named: (3, 2) float32.
y = tf.ones((3, 2), dtype=tf.float32)
b = tf.reverse_sequence(input=y, seq_lengths=lengths, seq_axis=0, batch_axis=1)
assert b.shape == (3, 2) and b.dtype == tf.float32
consume_float(b)

# A CRF-decode tail: int32 tags built by a concat, then reversed -> (2, 3) int32.
scores = tf.ones((2, 4), dtype=tf.float32)
first = tf.expand_dims(tf.cast(tf.argmax(scores, axis=1), dtype=tf.int32), axis=-1)
rest = tf.zeros((2, 2), dtype=tf.int32)
tags = tf.concat([first, rest], axis=1)
c = tf.reverse_sequence(tags, lengths, seq_axis=1)
assert c.shape == (2, 3) and c.dtype == tf.int32
consume_tags(c)

# One sink fed float32 logits by one caller and reversed int32 tags by another: both arms reach it.
consume_mixed(tf.ones((2, 3), dtype=tf.float32))
consume_mixed(c)
