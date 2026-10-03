import tensorflow as tf


class _Holder:
    def make(self):
        return tf.ones((7, 7))


# Shadows the submodule `shadow/mod.py`: Python's `from shadow import mod` takes this attribute of
# the package and never loads the submodule.
mod = _Holder()
