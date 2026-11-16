# Test for https://github.com/wala/ML/issues/1275: chains of binary operators in which each
# operator's operands are the results of the two operators before it, so every result is an operand
# of two later operators.
import numpy as np
import tensorflow as tf


def consume_tf(x):
    pass


def consume_np(x):
    pass


def consume_mixed(x):
    pass


def tf_chain():
    v0 = tf.ones([2, 3])
    v1 = v0 + 1.0
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
    v18 = v16 + v17
    v19 = v17 + v18
    v20 = v18 + v19
    v21 = v19 + v20
    v22 = v20 + v21
    v23 = v21 + v22
    v24 = v22 + v23
    v25 = v23 + v24
    v26 = v24 + v25
    v27 = v25 + v26
    v28 = v26 + v27
    v29 = v27 + v28
    v30 = v28 + v29
    v31 = v29 + v30
    assert v31.shape == (2, 3) and v31.dtype == tf.float32
    consume_tf(v31)


def np_chain():
    a0 = np.ones((2, 3), dtype=np.float32)
    a1 = a0 + a0
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
    assert isinstance(a31, np.ndarray)
    assert a31.shape == (2, 3) and a31.dtype == np.float32
    consume_np(a31)


def mixed_chain():
    m0 = np.ones((2, 3), dtype=np.float32)
    m1 = m0 + m0
    m2 = m0 + m1
    m3 = m1 + m2
    m4 = m2 + m3
    m5 = m3 + m4
    m6 = m4 + m5
    m7 = m5 + m6
    m8 = m6 + m7
    m9 = m7 + m8
    m10 = m8 + m9
    m11 = m9 + m10
    m12 = m10 + m11
    m13 = m11 + m12
    m14 = m12 + m13
    m15 = m13 + m14
    assert isinstance(m15, np.ndarray)
    # Still an array: no tensor has entered the chain.
    consume_np(m15)
    m16 = m14 + tf.ones([2, 3])
    m17 = m15 + m16
    m18 = m16 + m17
    m19 = m17 + m18
    m20 = m18 + m19
    m21 = m19 + m20
    m22 = m20 + m21
    m23 = m21 + m22
    m24 = m22 + m23
    m25 = m23 + m24
    m26 = m24 + m25
    m27 = m25 + m26
    m28 = m26 + m27
    m29 = m27 + m28
    m30 = m28 + m29
    m31 = m29 + m30
    assert tf.is_tensor(m31)
    assert m31.shape == (2, 3) and m31.dtype == tf.float32
    # A tensor: every operator from the tensor's on has a tensor operand.
    consume_mixed(m31)


tf_chain()
np_chain()
mixed_chain()
