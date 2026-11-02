# A function the call graph does not reach may still run. Its `update` on a dict read off a
# subscript may add an element to any dict.


def consume(c):
    pass


def merge(tables, key, entries):
    tables[key].update(entries)


c = {"a": 1, "b": 2}
assert len(c) == 2 and c["a"] == 1
consume(c)
