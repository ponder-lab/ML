import tensorflow as tf


def consume(x):
    pass


def never_called():
    # A foldable constant binary expression that raises at evaluation time. The constant folder
    # evaluates it in the embedded interpreter during class-hierarchy construction; division by zero
    # raises `ZeroDivisionError` (the NLPGNN case was a `NameError` on a free name -- both are
    # eval-time `PyException`s, not parse errors). Folding must skip an eval-time error and leave the
    # expression symbolic rather than aborting the whole hierarchy. The function is never called, so
    # the module still runs under `python3.10`. See wala/ML#640.
    return 1 / 0


t = tf.constant([1, 2, 3])
assert t.shape == (3,)
assert t.dtype == tf.int32
consume(t)
