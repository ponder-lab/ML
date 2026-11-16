# A dataset built by transformations that each apply only on a condition, so the receiver of each
# is the result of any of the earlier ones (wala/ML#1272). The batch at the end reads the size of
# every dataset it may be a batch of.
import random

import tensorflow as tf


def double(x):
    return x * 2


def consume(x):
    assert x.shape == (4, 3)
    assert x.dtype == tf.float32


def main():
    dataset = tf.data.Dataset.from_tensor_slices(tf.ones([8, 3]))
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
    if random.random() > 0.5:
        dataset = dataset.shuffle(8)
    if random.random() > 0.5:
        dataset = dataset.map(double)
    if random.random() > 0.5:
        dataset = dataset.take(8)
    for x in dataset.batch(4):
        consume(x)


main()
