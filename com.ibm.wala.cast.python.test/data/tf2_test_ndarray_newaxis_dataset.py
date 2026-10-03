# Test https://github.com/wala/ML/issues/1009: arithmetic on an array a Keras dataset loader returns
# is an array, as on any NumPy array, so the scaled and expanded images reach a dataset typed.
import tensorflow as tf


def consume(x):
    return x


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train[..., tf.newaxis]
assert x_train.shape == (60000, 28, 28, 1)
ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
for images, labels in ds.take(1):
    assert images.shape == (32, 28, 28, 1) and images.dtype == tf.float64
    consume(images)
