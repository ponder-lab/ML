import tensorflow as tf


def add(a, b):
    return a + a


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
dataset = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
)

for images, labels in dataset:
    assert isinstance(images, tf.Tensor)
    assert isinstance(labels, tf.Tensor)
    assert images.shape == (32, 28, 28)
    assert labels.shape == (32,)
    assert images.dtype == tf.uint8
    assert labels.dtype == tf.uint8
    c = add(images, labels)
