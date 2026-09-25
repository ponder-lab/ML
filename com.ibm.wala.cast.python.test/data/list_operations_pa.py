# Pointer-analysis witness for wala/ML#960: `Block.call` is reached ONLY through the repeated
# list, so its `past` parameter carries the None element iff repetition carries elements; the
# `sink` parameter carries both concatenated elements iff concatenation does.
class Block:
    def call(self, x, past=None):
        if past is not None:
            return x + past
        return x


def sink(v):
    return v


blocks = [Block(), Block()]
pasts = [None] * len(blocks)
for block, past in zip(blocks, pasts):
    block.call(1, past=past)

parts = [1] + [2]
for p in parts:
    sink(p)


def rep(xs, n):
    return xs * n


def sink2(v):
    return v


for q in rep([7], 3):
    sink2(q)


def prepend_one(ys):
    return [1] + ys


def two():
    return [2]


def sink3(v):
    return v


for w in prepend_one(two()):
    sink3(w)
