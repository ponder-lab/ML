# A function returning subscripts of a tensor in a tuple, unpacked at the call site
# (wala/ML#1261). A subscript of a tensor has an empty points-to set, so the elements reach the
# caller only through the returned literal's element writes. The two elements differ in shape and
# dtype, so a read of the wrong index, or a join of both, is told apart.
import tensorflow as tf


def crop(x, y):
    s = tf.stack([x, x], axis=0)
    u = tf.stack([y, y], axis=0)
    return s[0], u[1]


def crop_one(x):
    s = tf.stack([x, x], axis=0)
    return s[0]


def consume_first(r):
    assert r.shape == (2, 3)
    assert r.dtype == tf.float32


def consume_second(r):
    assert r.shape == (4,)
    assert r.dtype == tf.int32


def consume_one(r):
    assert r.shape == (2, 3)
    assert r.dtype == tf.float32


def normalize_first(image):
    assert image.shape == (2, 3)
    assert image.dtype == tf.float32
    consume_first(image / 127.5)


def normalize_second(image):
    assert image.shape == (4,)
    assert image.dtype == tf.int32
    consume_second(image // 2)


def normalize_one(image):
    assert image.shape == (2, 3)
    assert image.dtype == tf.float32
    consume_one(image / 127.5)


x = tf.ones([2, 3])
y = tf.zeros([4], tf.int32)
a, b = crop(x, y)
normalize_first(a)
normalize_second(b)
normalize_one(crop_one(x))
