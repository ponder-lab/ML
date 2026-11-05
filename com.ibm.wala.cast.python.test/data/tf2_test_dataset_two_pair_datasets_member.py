# Test a loop over one of two datasets of pairs whose first members differ in shape: position 0 of
# the loop variable is either dataset's first member.
import os
import tempfile

import tensorflow as tf


path = os.path.join(tempfile.mkdtemp(), "records.tfrecord")
with tf.io.TFRecordWriter(path) as writer:
    for _ in range(4):
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "x": tf.train.Feature(
                        float_list=tf.train.FloatList(value=[1.0] * 4)
                    ),
                    "y": tf.train.Feature(float_list=tf.train.FloatList(value=[0.0])),
                    "z": tf.train.Feature(
                        float_list=tf.train.FloatList(value=[2.0] * 3)
                    ),
                }
            )
        )
        writer.write(example.SerializeToString())


FEATURES = {
    "x": tf.io.FixedLenFeature([4], tf.float32),
    "y": tf.io.FixedLenFeature([], tf.float32),
    "z": tf.io.FixedLenFeature([3], tf.float32),
}


def parse_pair(record):
    f = tf.io.parse_single_example(record, FEATURES)
    return f["x"], f["y"]


def parse_z(record):
    f = tf.io.parse_single_example(record, FEATURES)
    return f["z"], f["y"]


def consume_member(m):
    pass


pairs = tf.data.TFRecordDataset([path]).map(parse_pair)
tensors = tf.data.TFRecordDataset([path]).map(parse_z)
flag = len([path]) == 2
ds = pairs if flag else tensors
for e in ds:
    m = e[0]
    assert m.shape == (3,), m.shape
    consume_member(m)
print("ok")
