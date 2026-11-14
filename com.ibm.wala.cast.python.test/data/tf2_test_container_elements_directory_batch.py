# A directory iterator's batch is an `(images, labels)` pair whose two positions its summary fixes
# (wala/ML#1265).
import os
import tempfile

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, save_img


def write_images():
    root = tempfile.mkdtemp()
    for label in ("a", "b"):
        os.makedirs(os.path.join(root, label))
        for i in range(2):
            save_img(os.path.join(root, label, "%d.png" % i), np.zeros((4, 4, 3)))
    return root


def sink_batch(batch):
    assert isinstance(batch, tuple) and len(batch) == 2
    assert batch[0].shape[1:] == (112, 112, 3) and batch[0].dtype == np.float32
    assert batch[1].ndim == 2 and batch[1].dtype == np.float32


def step(batch):
    sink_batch(batch)
    images, labels = batch
    return tf.reduce_sum(images) + tf.reduce_sum(labels)


gen = ImageDataGenerator(rescale=1.0 / 255).flow_from_directory(
    write_images(), target_size=(112, 112), batch_size=8, class_mode="categorical"
)
step(next(iter(gen)))
