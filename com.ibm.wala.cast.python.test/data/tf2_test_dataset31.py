# From https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/data/Dataset#zip.

import tensorflow as tf


def f(a):
    assert isinstance(a, tuple)


def g1(a):
    assert isinstance(a, tf.Tensor)


def g2(a):
    assert isinstance(a, tf.Tensor)


def h(a):
    assert isinstance(a, tuple)


def i1(a):
    assert isinstance(a, tf.Tensor)


def i2(a):
    assert isinstance(a, tf.Tensor)


def j(a):
    assert isinstance(a, tuple)


def k1(a):
    assert isinstance(a, tf.Tensor)


def k2(a):
    assert isinstance(a, tf.Tensor)


def k3(a):
    assert isinstance(a, tf.Tensor)


def l(a):
    assert isinstance(a, tuple)


def m1(a):
    assert isinstance(a, tf.Tensor)


def m2(a):
    assert isinstance(a, tf.Tensor)


# The nested structure of the `datasets` argument determines the
# structure of elements in the resulting dataset.
a = tf.data.Dataset.range(1, 4)  # ==> [ 1, 2, 3 ]
b = tf.data.Dataset.range(4, 7)  # ==> [ 4, 5, 6 ]
ds = tf.data.Dataset.zip((a, b))

for element in ds:
    assert isinstance(element, tuple)
    assert len(element) == 2
    assert isinstance(element[0], tf.Tensor)
    assert isinstance(element[1], tf.Tensor)
    assert element[0].shape == ()
    assert element[1].shape == ()
    assert element[0].dtype == tf.int64
    assert element[1].dtype == tf.int64
    f(element)
    assert len(element) == 2
    arg = element[0]
    assert isinstance(arg, tf.Tensor)
    assert arg.shape == ()
    assert arg.dtype == tf.int64
    g1(arg)
    arg2 = element[1]
    assert isinstance(arg2, tf.Tensor)
    assert arg2.shape == ()
    assert arg2.dtype == tf.int64
    g2(arg2)

ds = tf.data.Dataset.zip((b, a))

for element in ds:
    assert isinstance(element, tuple)
    assert len(element) == 2
    assert isinstance(element[0], tf.Tensor)
    assert isinstance(element[1], tf.Tensor)
    assert element[0].shape == ()
    assert element[1].shape == ()
    assert element[0].dtype == tf.int64
    assert element[1].dtype == tf.int64
    h(element)
    assert len(element) == 2
    arg = element[0]
    assert isinstance(arg, tf.Tensor)
    assert arg.shape == ()
    assert arg.dtype == tf.int64
    i1(arg)
    arg2 = element[1]
    assert isinstance(arg2, tf.Tensor)
    assert arg2.shape == ()
    assert arg2.dtype == tf.int64
    i2(arg2)

# The `datasets` argument may contain an arbitrary number of datasets.
c = tf.data.Dataset.range(7, 13).batch(2)  # ==> [ [7, 8],
#       [9, 10],
#       [11, 12] ]
ds = tf.data.Dataset.zip((a, b, c))

for element in ds:
    assert isinstance(element, tuple)
    assert len(element) == 3
    assert isinstance(element[0], tf.Tensor)
    assert isinstance(element[1], tf.Tensor)
    assert isinstance(element[2], tf.Tensor)
    assert element[0].shape == ()
    assert element[1].shape == ()
    assert element[2].shape == (2,)
    assert element[0].dtype == tf.int64
    assert element[1].dtype == tf.int64
    assert element[2].dtype == tf.int64
    j(element)
    assert len(element) == 3
    k1(element[0])
    k2(element[1])
    k3(element[2])

# The number of elements in the resulting dataset is the same as
# the size of the smallest dataset in `datasets`.
d = tf.data.Dataset.range(13, 15)  # ==> [ 13, 14 ]
ds = tf.data.Dataset.zip((a, d))

for element in ds:
    assert isinstance(element, tuple)
    assert len(element) == 2
    assert isinstance(element[0], tf.Tensor)
    assert isinstance(element[1], tf.Tensor)
    assert element[0].shape == ()
    assert element[1].shape == ()
    assert element[0].dtype == tf.int64
    assert element[1].dtype == tf.int64
    l(element)
    assert len(element) == 2
    m1(element[0])
    m2(element[1])
