# From https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/data/Dataset#map.

import tensorflow as tf
from tensorflow.data import Dataset


def f(a):
    assert isinstance(a, tf.Tensor)


dataset = Dataset.range(1, 6)  # ==> [ 1, 2, 3, 4, 5 ]
dataset = dataset.map(lambda x: x + 1)

for element in dataset:
    f(element)
