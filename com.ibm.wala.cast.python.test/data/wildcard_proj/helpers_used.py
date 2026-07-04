# Helper module for the wildcard-import probe (wala/ML#684): `tf` is a module global here, and —
# unlike `helpers.py` — it is also read by a function in this module, which lexically exposes the
# binding to the nested scope.
import tensorflow as tf


def shape_list(x):
    return tf.shape(x)
