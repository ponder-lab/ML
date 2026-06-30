import tensorflow as tf


def consume(x):
    pass


# The gpt-2 shape (wala/ML#645): a VarLenFeature in a feature dict, parsed, subscripted, densified.
# Currently untyped: the SparseTensor's empty PTS does not survive the dict subscript (wala/ML#646).
features = {"t": tf.io.VarLenFeature(tf.int64)}
parsed = tf.io.parse_single_example(b"", features)
d = tf.sparse.to_dense(parsed["t"])
consume(d)
