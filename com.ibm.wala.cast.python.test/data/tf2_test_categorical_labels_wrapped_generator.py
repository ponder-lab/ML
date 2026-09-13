import os
import tempfile

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Two `flow_from_directory` generators of DIFFERENT class counts built through one wrapper function
# (wala/ML#920). Only the ten-class generator's labels reach a `CategoricalCrossentropy` call. The
# consumer constraint keys membership on the labels allocation of a `flow_from_directory` call site,
# and a wrapper is one site serving two generators, so the width fixed for one must not be read for
# the other: the five-class labels must stay unresolved.

IMG = 32
BATCH = 4


def consume_wrapped_consumed(x):
    pass


def consume_wrapped_unconsumed(x):
    pass


def make_dir(num_classes):
    d = tempfile.mkdtemp()
    for c in range(num_classes):
        os.makedirs(os.path.join(d, "c%d" % c))
        for i in range(2):
            img = np.random.randint(0, 255, size=(IMG, IMG, 3), dtype=np.uint8)
            tf.keras.preprocessing.image.save_img(
                os.path.join(d, "c%d" % c, "%d.png" % i), img
            )
    return d


def flow(num_classes):
    return ImageDataGenerator(rescale=1.0 / 255).flow_from_directory(
        make_dir(num_classes),
        target_size=(IMG, IMG),
        batch_size=BATCH,
        class_mode="categorical",
    )


inp = tf.keras.layers.Input(shape=(IMG, IMG, 3))
model10 = tf.keras.Model(
    inputs=inp,
    outputs=tf.keras.layers.Dense(10, activation="softmax")(
        tf.keras.layers.Flatten()(inp)
    ),
)
loss_object = tf.keras.losses.CategoricalCrossentropy(
    reduction=tf.keras.losses.Reduction.NONE
)

images10, labels10 = next(iter(flow(10)))
assert labels10.shape == (BATCH, 10)
loss_object(labels10, model10(images10))
consume_wrapped_consumed(labels10)

images5, labels5 = next(iter(flow(5)))
assert labels5.shape == (BATCH, 5)
consume_wrapped_unconsumed(labels5)
