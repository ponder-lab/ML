# Two callers of one function, one guarding its call by a constant comparison, which folds from
# the callers' own values, and one by the rank of its argument, which folds from a shape the
# analysis reads (wala/ML#863).
import tensorflow as tf


def consume(x):
    assert x.shape == (2, 3)


def constant_caller(x):
    n = 3
    if n > 2:
        consume(x)


def rank_caller(x):
    if len(x.shape) == 2:
        consume(x)


def main():
    x = tf.ones([2, 3])
    constant_caller(x)
    rank_caller(x)


main()
