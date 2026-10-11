# Fixture for scikit-learn's `train_test_split`.
#
# Splitting a list keeps its elements: each half is a new list holding some of the input's
# elements, so iterating a half, or subscripting it with a computed index, reads the arrays the
# comprehension built. Splitting an array keeps its row type but not its row count, which is a
# fraction of the input's.
import numpy as np
from sklearn.model_selection import train_test_split


def consume_element(x):
    pass


def consume_test_element(x):
    pass


def consume_array(x):
    pass


def consume_indexed(x):
    pass


items = [np.ones([2, 3], dtype=np.float32) for _ in range(10)]
train, test = train_test_split(items, test_size=0.2)
assert len(train) == 8 and len(test) == 2

for t in train:
    assert t.shape == (2, 3)
    consume_element(t)

for t in test:
    assert t.shape == (2, 3)
    consume_test_element(t)

indexed = [train[i] for i in range(len(train))]
for t in indexed:
    assert t.shape == (2, 3)
    consume_indexed(t)

rows = np.ones([10, 4], dtype=np.float32)
rows_train, rows_test = train_test_split(rows, test_size=0.2)
assert rows_train.shape == (8, 4)
consume_array(rows_train)
