import tensorflow as tf


def consume(x):
    pass


# Static-analysis-only fixture: a VarLenFeature is a parse spec, not a sparse tensor, so
# tf.sparse.to_dense(vf) is not valid at TF runtime and this file does not run to completion. It
# exercises only Ariadne's modeling of the VarLenFeature -> SparseTensor -> to_dense path directly
# (the runnable, realistic form is tf2_test_parse_single_example.py), so it carries no runtime
# asserts. The modeled dtype propagates (int64).
vf = tf.io.VarLenFeature(tf.int64)
d = tf.sparse.to_dense(vf)
consume(d)
