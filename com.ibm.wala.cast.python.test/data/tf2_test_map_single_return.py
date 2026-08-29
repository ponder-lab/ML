import os
import tempfile

import tensorflow as tf


def consume_direct(x):
    pass


def consume_take(x):
    pass


def parse_example(serialized):
    data_fields = {"inputs": tf.io.VarLenFeature(tf.int64)}
    parsed = tf.io.parse_single_example(serialized, data_fields)
    return tf.cast(tf.sparse.to_dense(parsed["inputs"]), tf.int32)


# The wala/ML#848 reproduction: a mapped callback with a SINGLE tensor return (no tuple), whose
# element is consumed whole. The passing tuple-return fixture (tf2_test_tfrecord_map.py) reaches
# its elements through destructured per-index reads; this one exercises the whole-element route,
# once through the `take` pass-through and once directly off the mapped dataset, so the two
# consumers separate the pass-through inheritance from the plain mapped element.
with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "tf2_test_map_single_return.tfrecord")
    with tf.io.TFRecordWriter(path) as writer:
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "inputs": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[1, 2, 3])
                    ),
                }
            )
        )
        writer.write(example.SerializeToString())

    dataset = tf.data.TFRecordDataset(path).map(parse_example)

    for element in dataset:
        # Runtime shape is the concrete (3,) from the record; the static analysis recovers the
        # rank-1 dynamic (?,), since a VarLenFeature's element count is lost across serialization.
        assert element.shape == (3,), element.shape
        assert element.dtype == tf.int32, element.dtype
        consume_direct(element)

    for element in dataset.take(1):
        assert element.shape == (3,), element.shape
        assert element.dtype == tf.int32, element.dtype
        consume_take(element)
