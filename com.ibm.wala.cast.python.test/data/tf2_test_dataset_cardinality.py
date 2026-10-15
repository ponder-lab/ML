# Test for https://github.com/wala/ML/issues/1072.
import tensorflow as tf


def cardinality_sink(x):
    assert x.shape == ()
    assert x.dtype == tf.int64
    return x


def comparison_sink(x):
    assert x.shape == ()
    assert x.dtype == tf.bool
    return x


def is_infinite(dataset):
    cardinality_sink(dataset.cardinality())
    return comparison_sink(dataset.cardinality() == tf.data.INFINITE_CARDINALITY)


assert is_infinite(tf.data.Dataset.range(4).repeat())
