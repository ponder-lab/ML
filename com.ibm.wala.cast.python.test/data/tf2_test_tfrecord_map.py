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
# TFRecordDataset is now a chainable dataset, so .map resolves and the parsed targets type to
# (?,) int32. Static-analysis-only (no real tfrecord at runtime).
ds = tf.data.TFRecordDataset("x").map(parse_example)
for inputs, targets in ds:
    consume(targets)
