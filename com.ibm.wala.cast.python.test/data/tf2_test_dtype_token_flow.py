# The inter-procedural builtin dtype token (the NLPGNN graph-reader pattern): the builtin
# `int` flows as an argument into a reader helper whose `dtype` parameter reaches `np.array`,
# and the declaration must resolve through the call edge.
import numpy as np


def consume(a):
    pass


def read_raw(src, dtype=None):
    return np.array(src, dtype=dtype)


def consume2(b):
    pass


class Loader:
    def read_file(self, src, dtype=None):
        return self.read_raw_text(src, seq=",", dtype=dtype)

    def read_raw_text(self, src, seq=None, start=0, end=None, dtype=None):
        rows = [[dtype(x) for x in line.split(seq)[start:end]] for line in src]
        return np.array(rows, dtype=dtype)


edge_index = read_raw([1, 2, 3], dtype=int) - 1
assert edge_index.dtype == np.int64
assert edge_index.shape == (3,)
consume(edge_index)

loader = Loader()
batch = loader.read_file(["1,2", "3,4"], dtype=int) - 1
assert batch.dtype == np.int64
assert batch.shape == (2, 2)  # statically content-dependent
consume2(batch)
