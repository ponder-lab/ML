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


def consume_field(x):
    pass


def consume_mode(x):
    pass


def consume_tuple(x):
    pass


def consume_arg(x):
    pass


def pick_by_arg(flag):
    # The flag passed as an argument: the call-site binding rule must decline it as well.
    if flag:
        return tf.ones([2, 3])
    else:
        return tf.ones([4], dtype=tf.int32)


class Holder:
    def __init__(self, wide):
        self.wide = wide

    def pick(self):
        # The flag stored on an attribute: the field rule must decline it as well.
        if self.wide:
            return tf.ones([2, 3])
        else:
            return tf.ones([4], dtype=tf.int32)


@click.command()
@click.option("--wide", default=False)
@click.option("--width", default=6)
@click.option("--mode", default="narrow")
@click.option("--dims", default=(2, 3), nargs=2, type=int)
def main(wide, width, mode, dims):
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

    # The flag read back from an attribute decides nothing either.
    h = Holder(wide)
    f = h.pick()
    assert f.shape == (4,)
    consume_field(f)

    # The flag passed to a function decides nothing at that call site either.
    g = pick_by_arg(wide)
    assert g.shape == (4,)
    consume_arg(g)

    # A string default: `mode == "wide"` decides nothing, and the string still dispatches.
    if mode.lower() == "wide":
        m = tf.ones([2])
    else:
        m = tf.ones([3], dtype=tf.int32)
    assert m.shape == (3,)
    consume_mode(m)

    # A tuple default is no constant; it passes through as it is and reads as a shape.
    t = tf.ones(dims)
    assert t.shape == (2, 3)
    consume_tuple(t)


if __name__ == "__main__":
    main()
