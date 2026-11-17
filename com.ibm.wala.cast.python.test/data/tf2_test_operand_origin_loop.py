# Test for https://github.com/wala/ML/issues/1275: the chains of
# tf2_test_operand_origin_chain.py in a loop that carries their first two values, so every result
# is an operand of two later operators and reaches itself.
import numpy as np
import tensorflow as tf


def consume_tf(x):
    pass


def consume_np(x):
    pass


def tf_loop():
    v0 = tf.ones([2, 3])
    v1 = v0 + 1.0
    for _ in range(2):
        v2 = v0 + v1
        v3 = v1 + v2
        v4 = v2 + v3
        v5 = v3 + v4
        v6 = v4 + v5
        v7 = v5 + v6
        v8 = v6 + v7
        v9 = v7 + v8
        v10 = v8 + v9
        v11 = v9 + v10
        v12 = v10 + v11
        v13 = v11 + v12
        v14 = v12 + v13
        v15 = v13 + v14
        v16 = v14 + v15
        v17 = v15 + v16
        v0 = v16 + v17
        v1 = v17 + v0
    assert v17.shape == (2, 3) and v17.dtype == tf.float32
    consume_tf(v17)


def np_loop():
    a0 = np.ones((2, 3), dtype=np.float32)
    a1 = a0 + a0
    for _ in range(2):
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
        a0 = a16 + a17
        a1 = a17 + a0
    assert isinstance(a17, np.ndarray) and a17.shape == (2, 3)
    consume_np(a17)


tf_loop()
np_loop()
