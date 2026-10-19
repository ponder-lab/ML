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


def experimental_sink(x):
    assert x.shape == ()
    assert x.dtype == tf.int64
    return x


def unknown_comparison_sink(x):
    assert x.shape == ()
    assert x.dtype == tf.bool
    return x


def is_infinite(dataset):
    cardinality_sink(dataset.cardinality())
    return comparison_sink(dataset.cardinality() == tf.data.INFINITE_CARDINALITY)


def is_unknown(dataset):
    experimental_sink(tf.data.experimental.cardinality(dataset))
    return unknown_comparison_sink(
        tf.data.experimental.cardinality(dataset) == tf.data.UNKNOWN_CARDINALITY
    )


assert is_infinite(tf.data.Dataset.range(4).repeat())
assert is_unknown(tf.data.Dataset.range(4).filter(lambda x: x > 1))
