# Test https://github.com/wala/ML/issues/1190: the element of `Dataset.enumerate()` is a tuple of an
# int64 scalar index and the receiver's element, whose members keep their own types.
import tensorflow as tf


def consume_flat_index(i):
    pass


def consume_flat_x(x):
    pass


def consume_pair_index(i):
    pass


def consume_pair_x(x):
    pass


def consume_pair_y(y):
    pass


def consume_pair_element(e):
    pass


def consume_batched_index(i):
    pass


def consume_batched_x(x):
    pass


def consume_batched_y(y):
    pass


def consume_dict_index(i):
    pass


def consume_dict_a(a):
    pass


def consume_dict_b(b):
    pass


def pair_dataset():
    return tf.data.Dataset.from_tensor_slices(
        (tf.ones([6, 4]), tf.zeros([6, 2], dtype=tf.int32))
    )


def flat():
    for i, x in tf.data.Dataset.from_tensor_slices(tf.ones([5, 3])).enumerate():
        assert i.shape == () and i.dtype == tf.int64
        assert x.shape == (3,) and x.dtype == tf.float32
        consume_flat_index(i)
        consume_flat_x(x)


def pair():
    for i, (x, y) in pair_dataset().enumerate():
        assert i.shape == () and i.dtype == tf.int64
        assert x.shape == (4,) and x.dtype == tf.float32
        assert y.shape == (2,) and y.dtype == tf.int32
        consume_pair_index(i)
        consume_pair_x(x)
        consume_pair_y(y)


def pair_element():
    for i, e in pair_dataset().enumerate():
        assert isinstance(e, tuple) and len(e) == 2
        consume_pair_element(e)


def batched():
    for i, (x, y) in pair_dataset().enumerate().batch(2):
        assert i.shape == (2,) and i.dtype == tf.int64
        assert x.shape == (2, 4) and x.dtype == tf.float32
        assert y.shape == (2, 2) and y.dtype == tf.int32
        consume_batched_index(i)
        consume_batched_x(x)
        consume_batched_y(y)


def dict_element():
    dataset = tf.data.Dataset.from_tensor_slices(
        {"a": tf.ones([5, 3]), "b": tf.zeros([5], dtype=tf.int32)}
    )
    for i, d in dataset.enumerate():
        assert i.shape == () and i.dtype == tf.int64
        a = d["a"]
        assert a.shape == (3,) and a.dtype == tf.float32
        b = d["b"]
        assert b.shape == () and b.dtype == tf.int32
        consume_dict_index(i)
        consume_dict_a(a)
        consume_dict_b(b)


flat()
pair()
pair_element()
batched()
dict_element()
