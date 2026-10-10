# An array's `reshape` given its dimensions as separate integers, `x.reshape(2, 3)`, as well as one
# tuple or one integer. NumPy reads every positional integer as a dimension.
import json

import numpy as np


def consume_two(t):
    assert t.shape == (2, 3) and t.dtype == np.int64, (t.shape, t.dtype)


def consume_three(t):
    assert t.shape == (1, 2, 3) and t.dtype == np.int64, (t.shape, t.dtype)


def consume_row(t):
    assert t.shape == (3,) and t.dtype == np.int64, (t.shape, t.dtype)


def consume_subscript(t):
    assert t.shape == (3,) and t.dtype == np.int64, (t.shape, t.dtype)


def consume_inferred(t):
    assert t.shape == (3, 2) and t.dtype == np.int64, (t.shape, t.dtype)


def consume_one(t):
    assert t.shape == (6,) and t.dtype == np.int64, (t.shape, t.dtype)


def consume_tuple(t):
    assert t.shape == (2, 3) and t.dtype == np.int64, (t.shape, t.dtype)


x = np.arange(6).reshape(2, 3)
consume_two(x)
consume_three(np.arange(6).reshape(1, 2, 3))
for row in x:
    consume_row(row)
consume_subscript(x[0])
consume_inferred(np.arange(6).reshape(-1, 2))
consume_one(np.arange(6).reshape(6))
consume_tuple(np.arange(6).reshape((2, 3)))


def consume_unread(t):
    assert t.shape == (2, 3) and t.dtype == np.int64, (t.shape, t.dtype)


def consume_starred(t):
    assert t.shape == (2, 3) and t.dtype == np.int64, (t.shape, t.dtype)


rows = int(json.loads("2"))
consume_unread(np.arange(6).reshape(rows, 3))
dims = [2, 3]
consume_starred(np.arange(6).reshape(*dims))


# Two calls to one helper reshaping differently: each call's rows take its own dimensions.
def reshaped(a, n, m):
    return a.reshape(n, m)


def consume_helper_rows_of_two_by_three(t):
    assert t.shape == (3,) and t.dtype == np.int64, (t.shape, t.dtype)


def consume_helper_rows_of_three_by_two(t):
    assert t.shape == (2,) and t.dtype == np.int64, (t.shape, t.dtype)


for row in reshaped(np.arange(6), 2, 3):
    consume_helper_rows_of_two_by_three(row)
for row in reshaped(np.arange(6), 3, 2):
    consume_helper_rows_of_three_by_two(row)


# The reshaped array read back out of a list, so its type comes from the reshape's own allocation.
def consume_held(t):
    assert t.shape == (2, 3) and t.dtype == np.int64, (t.shape, t.dtype)


held = [np.arange(6).reshape(2, 3)]
consume_held(held[0])


def held_reshape(n, m):
    return [np.arange(6).reshape(n, m)]


def consume_held_two_by_three(t):
    assert t.shape == (2, 3) and t.dtype == np.int64, (t.shape, t.dtype)


def consume_held_three_by_two(t):
    assert t.shape == (3, 2) and t.dtype == np.int64, (t.shape, t.dtype)


consume_held_two_by_three(held_reshape(2, 3)[0])
consume_held_three_by_two(held_reshape(3, 2)[0])


# A loss test's arrays, each built from a flat literal reshaped by separate integers bound to locals.
class LossTest:
    def _run(self, acts, expected_costs, expected_grads):
        assert acts.shape == (2, 4, 3, 3) and acts.dtype == np.float32, (
            acts.shape,
            acts.dtype,
        )
        assert expected_costs.shape == (2,), expected_costs.shape
        assert expected_grads.shape == (2, 4, 3, 3), expected_grads.shape

    def test_batches(self):
        B = 2
        T = 4
        U = 3
        V = 3
        acts = np.array(
            [
                0.00,
                0.37,
                0.74,
                0.11,
                0.48,
                0.85,
                0.22,
                0.59,
                0.96,
                0.33,
                0.70,
                0.07,
                0.44,
                0.81,
                0.18,
                0.55,
                0.92,
                0.29,
                0.66,
                0.03,
                0.40,
                0.77,
                0.14,
                0.51,
                0.88,
                0.25,
                0.62,
                0.99,
                0.36,
                0.73,
                0.10,
                0.47,
                0.84,
                0.21,
                0.58,
                0.95,
                0.32,
                0.69,
                0.06,
                0.43,
                0.80,
                0.17,
                0.54,
                0.91,
                0.28,
                0.65,
                0.02,
                0.39,
                0.76,
                0.13,
                0.50,
                0.87,
                0.24,
                0.61,
                0.98,
                0.35,
                0.72,
                0.09,
                0.46,
                0.83,
                0.20,
                0.57,
                0.94,
                0.31,
                0.68,
                0.05,
                0.42,
                0.79,
                0.16,
                0.53,
                0.90,
                0.27,
            ],
            dtype=np.float32,
        ).reshape(B, T, U, V)
        expected_costs = np.array([4.28065, 3.93844], dtype=np.float32)
        expected_grads = np.array(
            [
                0.00,
                0.37,
                0.74,
                0.11,
                0.48,
                0.85,
                0.22,
                0.59,
                0.96,
                0.33,
                0.70,
                0.07,
                0.44,
                0.81,
                0.18,
                0.55,
                0.92,
                0.29,
                0.66,
                0.03,
                0.40,
                0.77,
                0.14,
                0.51,
                0.88,
                0.25,
                0.62,
                0.99,
                0.36,
                0.73,
                0.10,
                0.47,
                0.84,
                0.21,
                0.58,
                0.95,
                0.32,
                0.69,
                0.06,
                0.43,
                0.80,
                0.17,
                0.54,
                0.91,
                0.28,
                0.65,
                0.02,
                0.39,
                0.76,
                0.13,
                0.50,
                0.87,
                0.24,
                0.61,
                0.98,
                0.35,
                0.72,
                0.09,
                0.46,
                0.83,
                0.20,
                0.57,
                0.94,
                0.31,
                0.68,
                0.05,
                0.42,
                0.79,
                0.16,
                0.53,
                0.90,
                0.27,
            ],
            dtype=np.float32,
        ).reshape(B, T, U, V)
        self._run(acts, expected_costs, expected_grads)


LossTest().test_batches()


# The reshaped array unpacked from a tuple a helper returns, called twice with different dimensions:
# each unpacked array reaches the reshape through the tuple's element, not as the call's own result.
def paired(n, m):
    return np.arange(6).reshape(n, m), 0


def consume_paired_two_by_three(t):
    assert t.shape == (2, 3) and t.dtype == np.int64, (t.shape, t.dtype)


def consume_paired_three_by_two(t):
    assert t.shape == (3, 2) and t.dtype == np.int64, (t.shape, t.dtype)


first, _ = paired(2, 3)
consume_paired_two_by_three(first)
second, _ = paired(3, 2)
consume_paired_three_by_two(second)
