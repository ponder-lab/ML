# Test a loop over a map whose function returns `(x, y)` on one call and a tensor on another: the
# loop variable is that tensor, and its position 0 is a row of it.
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


def parse(record, labelled):
    f = tf.io.parse_single_example(
        record,
        {
            "x": tf.io.FixedLenFeature([4], tf.float32),
            "y": tf.io.FixedLenFeature([], tf.float32),
            "z": tf.io.FixedLenFeature([3], tf.float32),
        },
    )
    if labelled:
        return f["x"], f["y"]
    return f["z"] * 2.0


def consume_member(m):
    pass


def consume_element(e):
    pass


labelled = tf.data.TFRecordDataset([path]).map(lambda r: parse(r, True))
unlabelled = tf.data.TFRecordDataset([path]).map(lambda r: parse(r, False))
for e in unlabelled:
    assert e.shape == (3,), e.shape
    consume_element(e)
    m = e[0]
    assert m.shape == (), m.shape
    consume_member(m)
for p in labelled:
    assert p[0].shape == (4,)
print("ok")
