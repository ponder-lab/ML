# A dataset built by transformations that each apply only on a condition, so the receiver of each
# is the result of any of the earlier ones, whose batch's element is read by a function that
# branches on a rank (wala/ML#1299). Deciding the branch reads the element, and so the size of
# every dataset the batch may be a batch of, while the fold of the rank is in progress.
import random

import tensorflow as tf


def double(x, y):
    return x * 2, y


def by_rank(t):
    if len(t.shape) == 2:
        return tf.zeros([5, 2])
    return tf.zeros([7])


def consume(r):
    assert r.shape == (5, 2)
    assert r.dtype == tf.float32


def main():
    dataset = tf.data.Dataset.from_tensor_slices((tf.ones([8, 3]), tf.zeros([8])))
    if random.random() > 0.5:
        dataset = dataset.shuffle(8)
    if random.random() > 0.5:
        dataset = dataset.map(double)
    if random.random() > 0.5:
        dataset = dataset.take(8)
    if random.random() > 0.5:
        dataset = dataset.shuffle(8)
    if random.random() > 0.5:
        dataset = dataset.map(double)
    if random.random() > 0.5:
        dataset = dataset.take(8)
    if random.random() > 0.5:
        dataset = dataset.shuffle(8)
    if random.random() > 0.5:
        dataset = dataset.map(double)
    if random.random() > 0.5:
        dataset = dataset.take(8)
    for x, y in dataset.batch(4):
        consume(by_rank(x))


main()
