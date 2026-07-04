# Package-qualified helper for the wala/ML#684 probe: the package has no `__init__.py` (a
# namespace package, like the subject's `custom/`), and `tf` is also read by a function in this
# module.
import tensorflow as tf


def shape_list3(x):
    return tf.shape(x)
