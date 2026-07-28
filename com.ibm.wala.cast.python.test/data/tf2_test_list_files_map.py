# A tf.data.Dataset.list_files dataset's elements are 0-D filename-string tensors, so a map
# callback's parameter types as {[] string} (the Pix2Pix load(image_file) corpus form).
import os
import tempfile

import tensorflow as tf

folder = tempfile.mkdtemp()
for name in ("a.txt", "b.txt"):
    with open(os.path.join(folder, name), "w") as handle:
        handle.write("x")


def load(image_file):
    assert image_file.dtype == tf.string
    assert image_file.shape.rank == 0
    return tf.strings.length(image_file)


def consume(a):
    pass


dataset = tf.data.Dataset.list_files(os.path.join(folder, "*.txt"))
mapped = dataset.map(load)
for element in dataset:
    assert element.dtype == tf.string
    assert element.shape.rank == 0
    consume(element)
