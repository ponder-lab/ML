# Witness for wala/ML#971: a @click.option default is materialized as the parameter's value, and
# the comparison fold then decides guards over it.
# Unlike a Python default, a click default is not the only binding the
# program's closed world admits: every real invocation may pass `--wide True`, so a guard over it
# must NOT prune the other arm. Runtime asserts describe the DEFAULT invocation only.
import click
import tensorflow as tf


def consume_merge(x):
    pass


def consume_call(x):
    pass


def consume_shape(x):
    pass


def consume_arith(x):
    pass


def consume_plain(x):
    pass


@click.command()
@click.option("--wide", default=False)
@click.option("--width", default=6)
def main(wide, width):
    if wide:
        x = tf.ones([2, 3])
    else:
        x = tf.ones([4], dtype=tf.int32)
    # Default invocation: the int32 arm. With `--wide True`: float32 (2, 3).
    assert x.shape == (4,)
    consume_merge(x)

    if wide:
        consume_call(tf.zeros([5]))

    y = tf.ones([width])
    assert y.shape == (6,)
    consume_shape(y)

    # The default flows through arithmetic into a shape.
    z = tf.ones([width * 2])
    assert z.shape == (12,)
    consume_arith(z)

    # Control for the guarded call above: the same allocation, unguarded.
    w = tf.zeros([5])
    assert w.dtype == tf.float32
    consume_plain(w)


if __name__ == "__main__":
    main()
