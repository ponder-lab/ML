import numpy as np
import tensorflow as tf

# `TensorFlow2.0-Examples`'s `2-Basical_Models/CNN.py` in miniature: the division changes the
# dtype (no `astype`, so the images become float64), the ellipsis subscript adds the channel
# axis, the datasets chain `shuffle` before `batch`, and the step functions receive their
# arguments from destructuring iteration inside an epoch loop. The train image count divides
# by the batch exactly; the test count leaves a partial batch.

x_train = np.ones((96, 28, 28), dtype=np.uint8)
y_train = np.ones((96,), dtype=np.uint8)
x_test = np.ones((48, 28, 28), dtype=np.uint8)
y_test = np.ones((48,), dtype=np.uint8)

x_train, x_test = x_train / 255.0, x_test / 255.0

# Add a channels dimension.
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

train_ds = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)


@tf.function
def train_step(images, labels):
    pass


@tf.function
def test_step(images, labels):
    pass


EPOCHS = 2

for epoch in range(EPOCHS):
    for images, labels in train_ds:
        assert images.shape == (32, 28, 28, 1), images.shape
        assert images.dtype == tf.float64, images.dtype
        assert labels.shape == (32,), labels.shape
        assert labels.dtype == tf.uint8, labels.dtype
        train_step(images, labels)

    for test_images, test_labels in test_ds:
        assert test_images.shape in (
            (32, 28, 28, 1),
            (16, 28, 28, 1),
        ), test_images.shape
        assert test_images.dtype == tf.float64, test_images.dtype
        assert test_labels.shape in ((32,), (16,)), test_labels.shape
        assert test_labels.dtype == tf.uint8, test_labels.dtype
        test_step(test_images, test_labels)
