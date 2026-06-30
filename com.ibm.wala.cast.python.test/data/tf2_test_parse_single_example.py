import tensorflow as tf


def consume(x):
    pass


# The gpt-2 shape (wala/ML#645): a VarLenFeature in a feature dict, parsed, subscripted, densified.
# Currently untyped: the SparseTensor's empty PTS does not survive the dict subscript (wala/ML#646).
example = tf.train.Example(
    features=tf.train.Features(
        feature={"t": tf.train.Feature(int64_list=tf.train.Int64List(value=[1, 2]))}
    )
).SerializeToString()
features = {"t": tf.io.VarLenFeature(tf.int64)}
parsed = tf.io.parse_single_example(example, features)
d = tf.sparse.to_dense(parsed["t"])
assert d.shape == (2,)
assert d.dtype == tf.int64
consume(d)
