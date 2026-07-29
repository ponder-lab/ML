# The Pix2Pix corpus form (wala/ML#801 residual probe): list_files -> shuffle -> map with
# num_parallel_calls; the element type must survive the shuffle hop into the callback.
import os
import tempfile

import tensorflow as tf

folder = tempfile.mkdtemp()
for name in ("a.txt", "b.txt"):
    with open(os.path.join(folder, name), "w") as handle:
        handle.write("x")


def load_image_train(image_file):
    assert image_file.dtype == tf.string
    assert image_file.shape.rank == 0
    return tf.strings.length(image_file)


train_dataset = tf.data.Dataset.list_files(os.path.join(folder, "*.txt"))
train_dataset = train_dataset.shuffle(400)
train_dataset = train_dataset.map(
    load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE
)
for length in train_dataset:
    assert length.dtype == tf.int32
