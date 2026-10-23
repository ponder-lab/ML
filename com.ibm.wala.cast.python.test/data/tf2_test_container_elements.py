# Test `ContainerElements`: each sink receives one container, whose length, element keys and
# `None` elements the analysis reports.
import tensorflow as tf


def sink_tuple(c):
    assert len(c) == 2


def sink_list(c):
    assert len(c) == 3


def sink_appended(c):
    assert len(c) == 2


def sink_comprehension(c):
    assert len(c) == 3


def sink_dict(c):
    assert len(c) == 2


def sink_starred(c):
    assert len(c) == 3


def sink_extended(c):
    assert len(c) == 3


def sink_concatenated(c):
    assert len(c) == 3


def sink_repeated(c):
    assert len(c) == 4


def sink_with_none(c):
    assert len(c) == 2 and c[1] is None


def sink_without_none(c):
    assert len(c) == 2 and c[1] is not None


def sink_slice_assigned(c):
    assert len(c) == 3


def sink_augmented(c):
    assert len(c) == 2


def sink_counted(c):
    assert len(c) == 2


def mutate(l):
    l.insert(0, y)


def sink_mutated_elsewhere(c):
    assert len(c) == 2


def sink_ints(c):
    assert len(c) == 3


def sink_nested(c):
    assert len(c) == 2 and len(c[1]) == 2


x = tf.ones([2, 3])
y = tf.zeros([4], tf.int32)

sink_tuple((x, y))
sink_list([x, y, x])

appended = [x]
appended.append(y)
sink_appended(appended)

sink_comprehension([tf.ones([i + 1]) for i in range(3)])
sink_dict({"a": x, "b": y})

a = [x, y]
sink_starred([*a, x])

extended = [x]
extended.extend([y, x])
sink_extended(extended)

sink_concatenated([x] + [y, x])
sink_repeated([x, y] * 2)

sink_with_none((x, None))
sink_without_none((x, y))
sink_nested((x, (y, x)))

slice_assigned = [x, y]
slice_assigned[0:1] = [y, x]
sink_slice_assigned(slice_assigned)

augmented = [x]
augmented += [y]
sink_augmented(augmented)

counted = [x, y]
assert len(counted.copy()) == 2
sink_counted(counted)

mutated = [x]
mutate(mutated)
sink_mutated_elsewhere(mutated)

sink_ints([1, 2, 3])
