import os
import tempfile

import tensorflow as tf


def consume(x):
    pass


def parse_example(serialized):
    data_fields = {
        "inputs": tf.io.VarLenFeature(tf.int64),
        "targets": tf.io.VarLenFeature(tf.int64),
    }
    parsed = tf.io.parse_single_example(serialized, data_fields)
    inputs = tf.cast(tf.sparse.to_dense(parsed["inputs"]), tf.int32)
    targets = tf.cast(tf.sparse.to_dense(parsed["targets"]), tf.int32)
    return inputs, targets


# The gpt-2 input pipeline shape (wala/ML#618): a TFRecordDataset mapped by a VarLenFeature parser.
# TFRecordDataset is now a chainable dataset, so .map resolves. Write a real record so this runs.
path = os.path.join(tempfile.gettempdir(), "tf2_test_tfrecord_map.tfrecord")
with tf.io.TFRecordWriter(path) as writer:
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                "inputs": tf.train.Feature(int64_list=tf.train.Int64List(value=[1, 2])),
                "targets": tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[3, 4, 5])
                ),
            }
        )
    )
    writer.write(example.SerializeToString())

ds = tf.data.TFRecordDataset(path).map(parse_example)
for inputs, targets in ds:
    # Runtime shape is the concrete (3,) from the record; the static analysis recovers only the
    # rank-1 dynamic (?,), since the variable-length length is lost across serialize/parse. dtype is
    # int32 after the cast.
    assert targets.shape == (3,)
    assert targets.dtype == tf.int32
    consume(targets)
