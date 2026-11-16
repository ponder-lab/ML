# Test: mutually recursive functions, each returning its argument or a call to one of the next
# two, so their return values form one cycle. The generator dispatch of a result walked every
# simple path through the cycle (wala/ML#1274).
import tensorflow as tf


def f0(x, n):
    if n <= 0:
        return x
    if n % 2:
        return f1(x, n - 1)
    return f2(x, n - 1)


def f1(x, n):
    if n <= 0:
        return x
    if n % 2:
        return f2(x, n - 1)
    return f3(x, n - 1)


def f2(x, n):
    if n <= 0:
        return x
    if n % 2:
        return f3(x, n - 1)
    return f4(x, n - 1)


def f3(x, n):
    if n <= 0:
        return x
    if n % 2:
        return f4(x, n - 1)
    return f0(x, n - 1)


def f4(x, n):
    if n <= 0:
        return x
    if n % 2:
        return f0(x, n - 1)
    return f1(x, n - 1)


def consume(t):
    return t


def consume_other(t):
    return t


r = f0(tf.ones([2, 3]), 6)
assert r.shape == (2, 3), r.shape
assert r.dtype == tf.float32, r.dtype
consume(r)
s = f3(tf.zeros([4], tf.int32), 5)
assert s.shape == (4,), s.shape
assert s.dtype == tf.int32, s.dtype
consume_other(s)
