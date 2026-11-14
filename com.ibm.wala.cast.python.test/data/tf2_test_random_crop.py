# Test https://github.com/wala/ML/issues/1182: `tf.image.random_crop` returns a tensor with the shape
# its `size` argument gives and the dtype of the tensor it crops.
import tensorflow as tf


def consume(c):
    assert c.shape == (2, 3, 3) and c.dtype == tf.float32, c


def consume_keyword(c):
    assert c.shape == (1, 2) and c.dtype == tf.int32, c


consume(tf.image.random_crop(tf.ones([4, 4, 3]), [2, 3, 3]))
consume_keyword(tf.image.random_crop(value=tf.zeros([3, 3], tf.int32), size=[1, 2]))
