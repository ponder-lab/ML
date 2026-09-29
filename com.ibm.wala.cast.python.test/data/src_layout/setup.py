# A root-level script, as a project's `setup.py` is: only the project root's PYTHONPATH entry covers
# it, so analyzing it puts the root on the path alongside `src`.
import tensorflow as tf


def consume_setup(x):
    assert x.shape == (2, 2) and x.dtype == tf.float32


consume_setup(tf.ones((2, 2)))
