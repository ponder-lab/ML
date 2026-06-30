import tensorflow as tf


def consume(x):
    pass


sp = tf.sparse.SparseTensor(indices=[[0, 0], [1, 1]], values=[1, 2], dense_shape=[2, 2])
features = {"t": sp}
y = features["t"]
consume(y)
