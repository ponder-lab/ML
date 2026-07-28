# Staged probes for the wala/ML#796 downstream chain: each consume observes the reader's
# int64 one hop deeper (tuple unpack, slice application, comprehension list, dynamic
# subscript, re-array), so a per-probe assertion localizes where the dtype state stops
# flowing.
import numpy as np


def consume_unpacked(a):
    pass


def consume_sliced(b):
    pass


def consume_listed(c):
    pass


def consume_dynamic(d):
    pass


def consume_rearrayed(e):
    pass


class Loader:
    def read_file(self, src, dtype=None):
        rows = [[dtype(x) for x in line.split(",")] for line in src]
        return np.array(rows, dtype=dtype)

    def read_data(self, src):
        edge_index = self.read_file(src, dtype=int) - 1
        return edge_index, 0


loader = Loader()
edge_index, _zero = loader.read_data(["1,2", "3,4", "5,6"])
assert edge_index.dtype == np.int64
consume_unpacked(edge_index)

sliced = edge_index[0:2]
assert sliced.dtype == np.int64
consume_sliced(sliced)

value = [(0, 1), (1, 2)]
listed = [edge_index[se[0] : se[1]] for se in value]
assert listed[0].dtype == np.int64
consume_listed(listed[0])

for j in [0, 1]:
    dyn = listed[j]
    assert dyn.dtype == np.int64
    consume_dynamic(dyn)
    rearr = np.array(dyn)
    assert rearr.dtype == np.int64
    consume_rearrayed(rearr)
