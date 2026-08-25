# The multi-call-site form of the flow_from_directory batch tuple (wala/ML#834): TWO generators
# from distinct call sites with different target sizes. The per-position provider's caller hop
# must attribute each site its OWN image shape (the cross-wired-parameter defect the wala/ML#834
# review guarded against), and the mirrored helper parameter layout keeps the positions valid
# when the hop cannot single out one parent frame. Analysis-only as written: executing it needs
# `images_a/` and `images_b/` directories with class subfolders, which are not shipped; a full
# batch at runtime yields `(4, 64, 64, 3)`/`(4, num_classes)` from the first generator and
# `(4, 96, 96, 3)`/`(4, num_classes)` from the second, both `float32`. The `np.unique` tuple at
# the bottom is a NON-batch tuple carrying tensor types on its fields, present so field-evidence
# consumers can verify they filter by the allocating summary.
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale=1.0 / 255)
generator_a = datagen.flow_from_directory(
    "images_a", target_size=(64, 64), batch_size=4, class_mode="categorical"
)
generator_b = datagen.flow_from_directory(
    "images_b", target_size=(96, 96), batch_size=4, class_mode="categorical"
)


def step_a(inputs):
    images, labels = inputs
    return tf.reduce_mean(images) + tf.reduce_mean(labels)


def step_b(inputs):
    images, labels = inputs
    return tf.reduce_mean(images) + tf.reduce_mean(labels)


step_a(next(iter(generator_a)))
step_b(next(iter(generator_b)))

labels = np.array([3, 1, 3, 2], dtype=int)
_, _, inverse = np.unique(labels, return_index=True, return_inverse=True)
