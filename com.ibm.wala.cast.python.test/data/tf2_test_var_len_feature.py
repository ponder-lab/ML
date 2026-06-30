import tensorflow as tf


def consume(x):
    pass


# tf.io.VarLenFeature parses to a sparse tensor; densify it and check the dtype propagates (int64).
vf = tf.io.VarLenFeature(tf.int64)
d = tf.sparse.to_dense(vf)
consume(d)
