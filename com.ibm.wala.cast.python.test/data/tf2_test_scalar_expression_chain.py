# Test for https://github.com/wala/ML/issues/1275: a chain of binary operators over Python numbers
# that `math` functions return, in which each operator's operands are the results of the two
# operators before it. Multiplying a tensor by the chain's last value keeps the tensor's shape.
import math
import tensorflow as tf


def consume(x):
    pass


def scale(t, n):
    a0 = math.sqrt(n)
    a1 = math.sqrt(n + 1.0)
    a2 = a0 + a1
    a3 = a1 + a2
    a4 = a2 + a3
    a5 = a3 + a4
    a6 = a4 + a5
    a7 = a5 + a6
    a8 = a6 + a7
    a9 = a7 + a8
    a10 = a8 + a9
    a11 = a9 + a10
    a12 = a10 + a11
    a13 = a11 + a12
    a14 = a12 + a13
    a15 = a13 + a14
    a16 = a14 + a15
    a17 = a15 + a16
    a18 = a16 + a17
    a19 = a17 + a18
    a20 = a18 + a19
    a21 = a19 + a20
    a22 = a20 + a21
    a23 = a21 + a22
    a24 = a22 + a23
    a25 = a23 + a24
    a26 = a24 + a25
    a27 = a25 + a26
    a28 = a26 + a27
    a29 = a27 + a28
    a30 = a28 + a29
    a31 = a29 + a30
    return t * a31


r = scale(tf.ones([2, 3]), 2.0)
assert r.shape == (2, 3) and r.dtype == tf.float32
consume(r)
