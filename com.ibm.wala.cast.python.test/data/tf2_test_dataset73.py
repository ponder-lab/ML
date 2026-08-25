# The `train_step` shape of
# https://github.com/YunYang1994/TensorFlow2.0-Examples/blob/299fd6689f242d0f647a96b8844e86325e9fcb46/7-Utils/multi_gpu_train.py:
# the `(images, labels)` batch tuple from `flow_from_directory` arrives as a parameter and is
# destructured inside the consuming function, so each position must carry its own type rather than
# the whole-batch union (wala/ML#830). Like `tf2_test_dataset19.py`, this cannot execute as shipped
# (`./mnist/train` is not present). The asserts document what the iterator yields at the numpy
# level: batches of numpy ndarrays whose batch extent varies (the final batch of an epoch is short
# when the image count is not a multiple of `batch_size`) — the varying axis the analysis reports
# as its dynamic batch dimension.

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = 112
BATCH_SIZE = 512


def consume_images(images):
    assert images.shape[1:] == (IMG_SIZE, IMG_SIZE, 3)
    assert images.shape[0] <= BATCH_SIZE
    assert images.dtype == np.float32


def consume_labels(labels):
    assert labels.ndim == 2
    assert labels.dtype == np.float32


def consume_sparse_labels(labels):
    assert labels.ndim == 1
    assert labels.dtype == np.float32


def train_step(inputs):
    images, labels = inputs
    consume_images(images)
    consume_labels(labels)


train_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    "./mnist/train",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
)

train_dataset = iter(train_generator)
train_step(next(train_dataset))

sparse_generator = train_datagen.flow_from_directory(
    "./mnist/train",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="sparse",
)

sparse_images, sparse_labels = next(iter(sparse_generator))
consume_sparse_labels(sparse_labels)
