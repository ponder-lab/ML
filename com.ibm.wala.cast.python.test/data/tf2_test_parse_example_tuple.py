import tensorflow as tf


def consume(x):
    pass


# Mirrors gpt-2's parse_example (tuple return + cast over a VarLenFeature-through-dict to_dense),
# but called DIRECTLY (no dataset.map): the recovered dynamic int64 propagates through to_dense, the
# int32 cast, the tuple return, and the destructuring, so targets types to int32. Isolates the
# dataset.map element-type layer as the sole remaining gap for the full gpt-2 subject (wala/ML#618).
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


example = tf.train.Example(
    features=tf.train.Features(
        feature={
            "inputs": tf.train.Feature(int64_list=tf.train.Int64List(value=[1, 2])),
            "targets": tf.train.Feature(int64_list=tf.train.Int64List(value=[3, 4, 5])),
        }
    )
).SerializeToString()
inp, tar = parse_example(example)
# The runtime shape is the concrete (3,) from the example; the static analysis recovers only the
# rank-1 dynamic (?,), since the variable-length feature's length is lost across the serialize/parse
# round-trip. The dtype is int32 after the cast.
assert tar.shape == (3,)
assert tar.dtype == tf.int32
consume(tar)
