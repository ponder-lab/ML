# The three-hop corpus form of the graph-reader pattern (wala/ML#796): `read_data` calls
# `read_file` with mixed dtype tokens (`int` for indices, `float` for attributes), forwarding
# through `read_raw_text` over a real file read, so the reader is context-shared between the
# two declarations.
import os
import tempfile

import numpy as np


def consume(a):
    pass


def consume2(b):
    pass


class Loader:
    def read_file(self, folder, name, dtype=None):
        return self.read_raw_text(
            os.path.join(folder, name + ".txt"), seq=",", dtype=dtype
        )

    def read_raw_text(self, path, seq=None, start=0, end=None, dtype=None):
        with open(path, "r") as rf:
            src = rf.read().split("\n")[:-1]
        rows = [[dtype(x) for x in line.split(seq)[start:end]] for line in src]
        return np.array(rows, dtype=dtype)

    def read_data(self, folder):
        edge_index = self.read_file(folder, "A", dtype=int) - 1
        node_attributes = self.read_file(folder, "attrs", dtype=float)
        return edge_index, node_attributes


folder = tempfile.mkdtemp()
with open(os.path.join(folder, "A.txt"), "w") as f:
    f.write("1,2\n2,3\n")
with open(os.path.join(folder, "attrs.txt"), "w") as f:
    f.write("0.5,1.5\n2.5,3.5\n")

edges, attrs = Loader().read_data(folder)
assert edges.dtype == np.int64
assert attrs.dtype == np.float64
consume(edges)
consume2(attrs)
