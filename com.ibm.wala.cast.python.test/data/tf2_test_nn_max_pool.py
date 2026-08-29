import numpy as np
import tensorflow as tf

# The function-form pooling op at three feeds (wala/ML#857): two literal arms pinning both padding
# modes of the window fold, and the destructured-tuple-element arm whose read has an implicit
# pointer key, the wala/ML#855 configuration.


def consume_same(x):
    pass


def consume_valid(x):
    pass


def consume_destructured(x):
    pass


lit = tf.constant(np.ones((2, 16, 16, 3), dtype=np.float32))
same = tf.nn.max_pool(lit, ksize=2, strides=2, padding="SAME")
assert same.shape == (2, 8, 8, 3), same.shape
assert same.dtype == tf.float32, same.dtype
consume_same(same)

lit2 = tf.constant(np.ones((2, 17, 17, 3), dtype=np.float32))
valid = tf.nn.max_pool(lit2, ksize=3, strides=2, padding="VALID")
assert valid.shape == (2, 8, 8, 3), valid.shape
consume_valid(valid)

rows = 8
imgs = tf.constant(np.ones((rows, 16, 16, 3), dtype=np.float32))
labels = tf.constant(np.ones((rows,), dtype=np.float32))

loaded = tf.data.Dataset.from_tensor_slices((imgs, labels)).batch(
    4, drop_remainder=True
)

for X_img, Y in loaded:
    mp = tf.nn.max_pool(X_img, ksize=2, strides=2, padding="SAME")
    assert mp.shape == (4, 8, 8, 3), mp.shape
    consume_destructured(mp)
