# As tf2_test_dataset_nested_leaf_reads.py, with one function shared by two maps, so the leaves
# each map returns are read through the other's as well: their readings form a cycle
# (wala/ML#1270).
import tensorflow as tf


def pair(f, l):
    return f, l


def indexed(i, element):
    return element


def consume(x):
    assert x.shape == (1, 2, 3)
    assert x.dtype == tf.float32


def consume_label(y):
    assert y.shape == (1, 2)
    assert y.dtype == tf.int32


def main():
    features = tf.ones([4, 3])
    labels = tf.zeros([4], dtype=tf.int32)
    dataset = (
        tf.data.Dataset.from_tensor_slices((features, labels))
        .map(pair)
        .batch(2)
        .map(pair)
        .padded_batch(1)
        .enumerate()
        .map(indexed)
    )
    for x, y in dataset:
        consume(x)
        consume(tf.identity(x))
        consume_label(y)


main()
