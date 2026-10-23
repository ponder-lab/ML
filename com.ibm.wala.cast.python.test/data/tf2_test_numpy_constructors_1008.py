# Test https://github.com/wala/ML/issues/1008: NumPy array constructors and reductions that a
# data-preparation prefix builds its arrays with carry their shape and NumPy dtype.
import numpy as np


def consume_outer(x):
    pass


def consume_append(x):
    pass


def consume_concatenate(x):
    pass


def consume_stack(x):
    pass


def consume_fancy(x):
    pass


def consume_normalized(x):
    pass


def consume_choice(x):
    pass


def consume_choice_scalar(x):
    pass


def consume_multivariate_normal(x):
    pass


def consume_rescaled(x):
    pass


def consume_mean(x):
    pass


def consume_mean_axis(x):
    pass


def consume_mean_keepdims(x):
    pass


def consume_std(x):
    pass


def consume_mean_dtype(x):
    pass


def consume_repeat(x):
    pass


def consume_diag(x):
    pass


def consume_choice_index(x):
    pass


def consume_permutation_index(x):
    pass


nt = int(8)
ns = int(5)
a1 = np.random.normal(-2, 0.5, ns)
p = np.append(np.arange(-nt / 2, 0), 0.5 * np.arange(0, nt / 2)[1:])
assert p.shape == (7,) and p.dtype == np.float64
consume_append(p)

X0 = np.outer(a1, 2 * p)
assert X0.shape == (5, 7) and X0.dtype == np.float64
consume_outer(X0)

X_train = np.concatenate((X0, X0), axis=0)
assert X_train.shape == (10, 7) and X_train.dtype == np.float64
consume_concatenate(X_train)

ii = np.random.permutation(ns * 2)
X_train = np.stack((X_train, X_train), axis=2)
assert X_train.shape == (10, 7, 2) and X_train.dtype == np.float64
consume_stack(X_train)

X_train = X_train[ii, :]
assert X_train.shape == (10, 7, 2) and X_train.dtype == np.float64
consume_fancy(X_train)

W_train = np.random.choice(2, X_train.shape, p=[0.2, 0.8])
assert W_train.shape == (10, 7, 2) and W_train.dtype == np.int64
consume_choice(W_train)

k = np.random.choice(3)
assert isinstance(k, int)
consume_choice_scalar(k)

for i in np.arange(X_train.shape[2]):
    X_train[:, :, i] = (X_train[:, :, i] - np.mean(X_train[:, :, i])) / np.std(
        X_train[:, :, i]
    )
assert X_train.shape == (10, 7, 2) and X_train.dtype == np.float64
consume_normalized(X_train)

m = np.repeat(-1, nt)
assert m.shape == (8,) and m.dtype == np.int64
consume_repeat(m)

c = np.diag(np.repeat(2, nt))
assert c.shape == (8, 8) and c.dtype == np.int64
consume_diag(c)

b1 = np.random.multivariate_normal(m, c, 4)
assert b1.shape == (4, 8) and b1.dtype == np.float64
consume_multivariate_normal(b1)

Y = np.concatenate((b1, b1), axis=0)
Y = (Y - np.mean(Y)) / np.std(Y)
assert Y.shape == (8, 8) and Y.dtype == np.float64
consume_rescaled(Y)

mean = np.mean(m)
assert np.ndim(mean) == 0 and mean.dtype == np.float64
consume_mean(mean)

mean_axis = np.mean(c, axis=0)
assert mean_axis.shape == (8,) and mean_axis.dtype == np.float64
consume_mean_axis(mean_axis)

mean_keepdims = np.mean(b1, axis=1, keepdims=True)
assert mean_keepdims.shape == (4, 1) and mean_keepdims.dtype == np.float64
consume_mean_keepdims(mean_keepdims)

mean_dtype = np.mean(m, dtype=np.dtype("float32"))
assert np.ndim(mean_dtype) == 0 and mean_dtype.dtype == np.float32
consume_mean_dtype(mean_dtype)

std = np.std(np.ones((3, 2), np.float32))
assert np.ndim(std) == 0 and std.dtype == np.float32
consume_std(std)

jj = np.random.choice(np.arange(10), 4, replace=False)
picked = X_train[jj,]
assert picked.shape == (4, 7, 2) and picked.dtype == np.float64
consume_choice_index(picked)

kk = np.random.permutation(10)
permuted = X_train[kk]
assert permuted.shape == (10, 7, 2) and permuted.dtype == np.float64
consume_permutation_index(permuted)


def consume_diag_offset(x):
    pass


def consume_diag_negative_offset(x):
    pass


def consume_diag_of_matrix(x):
    pass


def consume_diag_of_matrix_offset(x):
    pass


def consume_diag_unresolved_offset(x):
    pass


def consume_arange_stop(x):
    pass


def consume_arange_float_step(x):
    pass


def consume_arange_product(x):
    pass


def consume_arange_negated(x):
    pass


def consume_arange_global_float(x):
    pass


def consume_arange_unclassified(x):
    pass


def consume_mean_axis_none(x):
    pass


def consume_mean_axis_tuple(x):
    pass


def consume_mean_keepdims_all(x):
    pass


def consume_mean_negative_axis(x):
    pass


def consume_mean_float32(x):
    pass


def consume_mean_bool(x):
    pass


def consume_choice_list(x):
    pass


def consume_multivariate_normal_vector(x):
    pass


def consume_append_axis(x):
    pass


def consume_repeat_axis(x):
    pass


def consume_repeat_unresolved(x):
    pass


d1 = np.diag(np.ones(3), 1)
assert d1.shape == (4, 4) and d1.dtype == np.float64
consume_diag_offset(d1)

d2 = np.diag(np.ones(3), -2)
assert d2.shape == (5, 5) and d2.dtype == np.float64
consume_diag_negative_offset(d2)

d3 = np.diag(np.ones((3, 5)))
assert d3.shape == (3,) and d3.dtype == np.float64
consume_diag_of_matrix(d3)

d4 = np.diag(np.ones((3, 5)), 1)
assert d4.shape == (3,) and d4.dtype == np.float64
consume_diag_of_matrix_offset(d4)

d5 = np.diag(np.ones(3), np.random.randint(0, 2))
assert d5.shape[0] in (3, 4) and d5.dtype == np.float64
consume_diag_unresolved_offset(d5)

r1 = np.arange(5)
assert r1.shape == (5,) and r1.dtype == np.int64
consume_arange_stop(r1)

r2 = np.arange(0, 1, 0.25)
assert r2.shape == (4,) and r2.dtype == np.float64
consume_arange_float_step(r2)

k3 = int(3)
r3 = np.arange(k3 * 2)
assert r3.shape == (6,) and r3.dtype == np.int64
consume_arange_product(r3)

r4 = np.arange(-k3, 0)
assert r4.shape == (3,) and r4.dtype == np.int64
consume_arange_negated(r4)

g_float = 2.5


def arange_global():
    return np.arange(0, g_float)


r5 = arange_global()
assert r5.shape == (3,) and r5.dtype == np.float64
consume_arange_global_float(r5)

r6 = np.arange(np.random.randint(1, 3))
assert r6.ndim == 1 and r6.dtype == np.int64
consume_arange_unclassified(r6)

m1 = np.mean(c, axis=None)
assert np.ndim(m1) == 0 and m1.dtype == np.float64
consume_mean_axis_none(m1)

m2 = np.mean(c, axis=(0, 1))
assert np.ndim(m2) == 0 and m2.dtype == np.float64
consume_mean_axis_tuple(m2)

m3 = np.mean(c, keepdims=True)
assert m3.shape == (1, 1) and m3.dtype == np.float64
consume_mean_keepdims_all(m3)

m4 = np.mean(b1, axis=-1)
assert m4.shape == (4,) and m4.dtype == np.float64
consume_mean_negative_axis(m4)

m5 = np.mean(np.ones((2, 3), np.float32), axis=0)
assert m5.shape == (3,) and m5.dtype == np.float32
consume_mean_float32(m5)

m6 = np.mean(np.ones((2, 3), np.bool_))
assert np.ndim(m6) == 0 and m6.dtype == np.float64
consume_mean_bool(m6)

ch = np.random.choice([1, 2, 3], 2)
assert ch.shape == (2,) and ch.dtype == np.int64
consume_choice_list(ch)

v = np.random.multivariate_normal(m, c)
assert v.shape == (8,) and v.dtype == np.float64
consume_multivariate_normal_vector(v)

ap = np.append(c, c, axis=0)
assert ap.shape == (16, 8) and ap.dtype == np.int64
consume_append_axis(ap)

rp = np.repeat(c, 2, axis=0)
assert rp.shape == (16, 8) and rp.dtype == np.int64
consume_repeat_axis(rp)

ru = np.repeat(m, np.random.randint(1, 3))
assert ru.ndim == 1 and ru.dtype == np.int64
consume_repeat_unresolved(ru)
