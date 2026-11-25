# Test https://github.com/wala/ML/issues/1334: `x.set_shape(y.shape)` keeps the type that flows to
# `x`, since `set_shape` never changes the value it is called on and `y.shape` is no list or tuple
# literal whose elements give a shape.
import tensorflow as tf


def consume_logits(t):
    assert t.shape == (2, 4, 3, 3)
    assert t.dtype == tf.float32


def consume_labels(t):
    assert t.shape == (2, 2)
    assert t.dtype == tf.int32


def pin(logits_t, labels_t, logits, labels):
    logits_t.set_shape(logits.shape)
    labels_t.set_shape(labels.shape)
    consume_logits(logits_t)
    consume_labels(labels_t)


logits = tf.ones([2, 4, 3, 3])
labels = tf.ones([2, 2], tf.int32)
pin(logits, labels, logits, labels)
