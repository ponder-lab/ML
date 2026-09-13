# Mirrors `HardNegativeMining` from `deep_recommenders/keras/models/retrieval/sbcnm.py` together
# with its only driver, `tests/keras/test_sbcnm.py`, at the subject's shape (wala/ML#907): the
# `top_k` indices are computed inside a `Layer.call` reached through `__call__`, over an
# elementwise combination of the two call parameters and a module-level numpy scalar, with a
# tensor-valued `k`, and then passed into a module-level helper as `column_indices`. The driver
# is a parameterized test method (a pytest-shaped file, class and method, so that the analysis
# binds it as an entrypoint the way the subject's own test runner does) calling the layer twice,
# the second time on a numpy binop of its own arguments.
from typing import Tuple

import numpy as np
import parameterized
import tensorflow as tf

MAX_FLOAT = np.finfo(np.float32).max / 100.0


def _gather_elements_along_row(data: tf.Tensor, column_indices: tf.Tensor) -> tf.Tensor:
    with tf.control_dependencies(
        [tf.assert_equal(tf.shape(data)[0], tf.shape(column_indices)[0])]
    ):
        num_row = tf.shape(data)[0]
        num_column = tf.shape(data)[1]
        num_gathered = tf.shape(column_indices)[1]
        row_indices = tf.tile(tf.expand_dims(tf.range(num_row), -1), [1, num_gathered])
        flat_data = tf.reshape(data, [-1])
        flat_indices = tf.reshape(row_indices * num_column + column_indices, [-1])
        return tf.reshape(tf.gather(flat_data, flat_indices), [num_row, num_gathered])


class HardNegativeMining(tf.keras.layers.Layer):
    def __init__(self, num_hard_negatives: int, **kwargs):
        super(HardNegativeMining, self).__init__(**kwargs)
        self._num_hard_negatives = num_hard_negatives

    def call(self, logits: tf.Tensor, labels: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        num_sampled = tf.minimum(self._num_hard_negatives + 1, tf.shape(logits)[1])

        _, indices = tf.nn.top_k(
            logits + labels * MAX_FLOAT, k=num_sampled, sorted=False
        )

        logits = _gather_elements_along_row(logits, indices)
        labels = _gather_elements_along_row(labels, indices)

        return logits, labels


class TestSBCNM(tf.test.TestCase):
    @parameterized.parameters(3, 5, 10, 15)
    def test_hard_negative_mining(self, num_hard_negatives):
        logits_shape = (2, 20)
        rng = np.random.RandomState(42)

        logits = rng.uniform(size=logits_shape).astype(np.float32)
        labels = rng.permutation(np.eye(*logits_shape).T).T.astype(np.float32)

        out_logits, out_labels = HardNegativeMining(num_hard_negatives)(logits, labels)

        self.assertEqual(out_logits.shape[-1], num_hard_negatives + 1)

        logits = logits + labels * 1000.0

        out_logits, out_labels = HardNegativeMining(num_hard_negatives)(logits, labels)
        out_logits, out_labels = out_logits.numpy(), out_labels.numpy()
        # The gathered result has the indices' shape: rank 2, batch 2, k = num_hard_negatives + 1
        # (k is a runtime tensor, so that axis is `None` statically and concrete at run time).
        assert out_logits.shape == (2, num_hard_negatives + 1), out_logits.shape
        assert out_labels.shape == (2, num_hard_negatives + 1), out_labels.shape


if __name__ == "__main__":
    tf.test.main()
