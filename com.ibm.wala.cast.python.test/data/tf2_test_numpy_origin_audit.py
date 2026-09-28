# Test https://github.com/wala/ML/issues/980: every modelled numpy API produces an ndarray, so
# each result routed to `consume` must read the numpy origin.
import numpy as np


def consume(x):
    pass


m = np.ones((2, 3))
consume(np.array([[1.0, 2.0]]))
consume(np.ones((2, 3)))
consume(np.zeros((2, 3)))
consume(np.eye(3))
consume(np.arange(6))
consume(np.pad(m, 1))
consume(np.transpose(m))
consume(m.transpose())
consume(m.T)
consume(np.reshape(m, (3, 2)))
consume(m.reshape((3, 2)))
consume(m.astype(np.int32))
consume(np.cumsum(m))
consume(np.unique(np.array([1, 2, 2]), return_counts=True)[0])
consume(np.random.rand(2, 3))
consume(np.random.randn(2, 3))
consume(np.random.randint(0, 5, size=(2, 3)))
consume(np.random.uniform(size=(2, 3)))
consume(np.random.normal(size=(2, 3)))
consume(np.random.permutation(5))
consume(m[0])
consume(m + 1.0)

w = np.ones((2, 3))
for _ in range(2):
    w = w.T
consume(w)
