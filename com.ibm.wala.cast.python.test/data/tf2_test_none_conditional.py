# Test that `None` written as one arm of a conditional reaches a parameter as a `None` member, passed
# directly or forwarded, while a variable left unbound on one path does not.
import contextlib

import tensorflow as tf


def direct_sink(x, n):
    if n is None:
        return x
    return x * tf.cast(n, x.dtype)


def direct(x, aligned):
    return direct_sink(x, tf.constant(2) if aligned else None)


def forwarded_sink(x, n):
    if n is None:
        return x
    return x * tf.cast(n, x.dtype)


def forward(x, n):
    return forwarded_sink(x, n)


def forwarded(x, aligned):
    return forward(x, tf.constant(2) if aligned else None)


def statement_sink(x, n):
    if n is None:
        return x
    return x * tf.cast(n, x.dtype)


def statement(x, aligned):
    if aligned:
        n = tf.constant(2)
    else:
        n = None
    return statement_sink(x, n)


def both_sink(x, n):
    return x * tf.cast(n, x.dtype)


def both(x, aligned):
    return both_sink(x, tf.constant(2) if aligned else tf.constant(3))


def unbound_sink(x, n):
    return x * tf.cast(n, x.dtype)


def unbound(x, aligned):
    if aligned:
        n = tf.constant(2)
    return unbound_sink(x, n)


def asserted_sink(x, n):
    return x * tf.cast(n, x.dtype)


def asserted(x, n=None, start=None):
    if start is None:
        assert n is not None
    else:
        n = tf.constant(start)
    return asserted_sink(x, n)


def named_sink(x, n):
    if n is None:
        return x
    return x * tf.cast(n, x.dtype)


def named(x, aligned):
    n = tf.constant(2) if aligned else None
    return named_sink(x, n)


def caught_sink(x, n):
    if n is None:
        return x
    return x * tf.cast(n, x.dtype)


def caught(x, n=None):
    try:
        assert n is not None
    except AssertionError:
        pass
    return caught_sink(x, n)


def suppressed_sink(x, n):
    if n is None:
        return x
    return x * tf.cast(n, x.dtype)


def suppressed(x, n=None):
    with contextlib.suppress(AssertionError):
        assert n is not None
    return suppressed_sink(x, n)


def param_arm_sink(x, n):
    if n is None:
        return x
    return x * tf.cast(n, x.dtype)


def param_arm(x, n, aligned):
    return param_arm_sink(x, n if aligned else tf.constant(3))


def loop_first_sink(x, n):
    return x * tf.cast(n, x.dtype)


def loop_first(x, k):
    while k > 0:
        n = tf.constant(2)
        k -= 1
    return loop_first_sink(x, n)


def method_sink(x, n):
    if n is None:
        return x
    return x * tf.cast(n, x.dtype)


class Asserter:
    def apply(self, x, n=None, start=None):
        if start is None:

            # The caller passes `n` whenever it passes no `start`.
            assert n is not None
        else:
            n = tf.constant(start)
        return method_sink(x, n)


x = tf.constant([1.0, 2.0])

assert direct(x, True).shape == (2,)
assert direct(x, False).shape == (2,)

assert forwarded(x, True).shape == (2,)
assert forwarded(x, False).shape == (2,)

assert statement(x, True).shape == (2,)
assert statement(x, False).shape == (2,)

assert both(x, True).shape == (2,)
assert both(x, False).shape == (2,)

assert unbound(x, True).shape == (2,)
try:
    unbound(x, False)
    assert False
except UnboundLocalError:
    pass

assert asserted(x, tf.constant(2)).shape == (2,)
assert asserted(x, start=3).shape == (2,)

assert named(x, True).shape == (2,)
assert named(x, False).shape == (2,)

assert caught(x, tf.constant(2)).shape == (2,)
assert caught(x).shape == (2,)

assert suppressed(x, tf.constant(2)).shape == (2,)
assert suppressed(x).shape == (2,)

assert param_arm(x, None, True).shape == (2,)
assert param_arm(x, tf.constant(2), True).shape == (2,)
assert param_arm(x, None, False).shape == (2,)

assert loop_first(x, 1).shape == (2,)

asserter = Asserter()
assert asserter.apply(x, tf.constant(2)).shape == (2,)
assert asserter.apply(x, start=3).shape == (2,)

assert tf.constant(2).dtype == tf.int32
assert tf.constant(2).shape == ()
