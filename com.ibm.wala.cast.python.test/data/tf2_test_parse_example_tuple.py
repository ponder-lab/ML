import tensorflow as tf


def consume(x):
    pass


# Mirrors gpt-2's parse_example (tuple return + cast over a VarLenFeature-through-dict to_dense),
# but called DIRECTLY (no dataset.map): the recovered (?,) int64 propagates through to_dense, the
# int32 cast, the tuple return, and the destructuring, so targets types to (?,) int32. Isolates the
# dataset.map element-type layer as the sole remaining gap for the full gpt-2 subject (wala/ML#618).
# Static-analysis-only (a VarLenFeature is a parse spec, not a runtime sparse tensor).
def parse_example(serialized):
    data_fields = {
        "inputs": tf.io.VarLenFeature(tf.int64),
        "targets": tf.io.VarLenFeature(tf.int64),
    }
    parsed = tf.io.parse_single_example(serialized, data_fields)
    inputs = tf.sparse.to_dense(parsed["inputs"])
    targets = tf.sparse.to_dense(parsed["targets"])
    inputs = tf.cast(inputs, tf.int32)
    targets = tf.cast(targets, tf.int32)
    return inputs, targets


inp, tar = parse_example("")
consume(tar)
