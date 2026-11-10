# Test for wala/ML#782: `glob.glob` returns a list of file names, which `TFRecordDataset` reads as a
# string tensor of names.
import glob
import os
import tempfile

import tensorflow as tf


def input_fn(tf_records):
    assert isinstance(tf_records, list)
    return tf.data.TFRecordDataset(tf_records)


directory = tempfile.mkdtemp()
with tf.io.TFRecordWriter(os.path.join(directory, "a.tfrecord")) as writer:
    writer.write(b"record")

dataset = input_fn(glob.glob(os.path.join(directory, "*.tfrecord")))
for record in dataset:
    assert record.dtype == tf.string
