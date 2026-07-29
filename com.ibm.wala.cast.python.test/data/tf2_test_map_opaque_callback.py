# The wala/ML#803 guard: a map callback that transforms its element through an unmodeled op
# must yield an honest unknown element type, never the receiver's element type (the runtime
# dtype here is float32; the receiver's is string).
import os
import tempfile

import tensorflow as tf

folder = tempfile.mkdtemp()
for name in ("1.txt", "2.txt"):
    with open(os.path.join(folder, name), "w") as handle:
        handle.write("4.9")


def to_number(image_file):
    return tf.strings.to_number(tf.io.read_file(image_file))


def consume(a):
    pass


dataset = tf.data.Dataset.list_files(os.path.join(folder, "*.txt"))
mapped = dataset.map(to_number)
for element in mapped:
    assert element.dtype == tf.float32
    consume(element)
