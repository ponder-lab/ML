# Test https://github.com/wala/ML/issues/1009: an `np.reshape` result read through a tuple's element,
# assigned through a tuple and read as `tf.data.Dataset.from_tensor_slices((x, y))` reads its
# components, is typed by the reshape.
import numpy as np
import tensorflow as tf


def consume(labels):
    return labels


x = np.ones((64, 3), dtype=np.float32)
y = np.zeros((64, 1), dtype=np.uint8)
z = np.zeros((16, 1), dtype=np.uint8)
y, z = np.reshape(y, (-1)), np.reshape(z, (-1))
ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(32)
for batch_x, batch_y in ds.take(1):
    assert batch_y.shape == (32,) and batch_y.dtype == tf.uint8
    consume(batch_y)
