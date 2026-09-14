# Minimal reproduction for wala/ML#923: a rank assert on a tuple-unpacked op result, inside the body
# that produced it, when the op's argument read walks to the callers (k is a tensor).
import tensorflow as tf


def pick(scores, n):
    k = tf.minimum(n + 1, tf.shape(scores)[1])
    _, indices = tf.nn.top_k(scores, k=k, sorted=False)
    assert indices.shape.rank == 2 and indices.shape[0] == 2
    return indices


class TestPick:
    def test_pick(self):
        scores = tf.constant([[1.0, 3.0, 2.0, 5.0, 4.0], [5.0, 4.0, 3.0, 2.0, 1.0]])
        out = pick(scores, 2)
        assert out.shape == (2, 3)


TestPick().test_pick()
