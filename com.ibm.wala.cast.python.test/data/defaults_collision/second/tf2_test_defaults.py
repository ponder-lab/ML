# The sibling script of `first/tf2_test_defaults.py` for wala/ML#1337.
import tensorflow as tf


def consume2(t):
    assert t.shape == (7, 5)
    assert t.dtype == tf.float32


def main(verbose=False):
    if verbose:
        print("verbose")
    consume2(tf.ones([7, 5]))


if __name__ == "__main__":
    main()
