import tensorflow as tf


def consume(x):
    pass


# `tf.sparse.to_dense` of a SparseTensor: does Ariadne type the densified result? Probes the
# construct the gpt-2 `input_fn`'s `parse_example` relies on (wala/ML#618).
sp = tf.sparse.SparseTensor(indices=[[0, 0], [1, 1]], values=[1, 2], dense_shape=[2, 2])
d = tf.sparse.to_dense(sp)
consume(d)
