reached = []


def raises(x):
    raise ValueError(x)


def typed(x):
    reached.append("typed")


def typed_as(x):
    reached.append("typed_as")


def typed_tuple(x):
    reached.append("typed_tuple")


def broad(x):
    reached.append("broad")


def bare(x):
    reached.append("bare")


def nested_outer(x):
    reached.append("nested_outer")


def after(x):
    reached.append("after")


def raised_directly(x):
    reached.append("raised_directly")


def in_function(x):
    reached.append("in_function")


try:
    raises(1)
except ValueError:
    typed(1)

try:
    raises(2)
except ValueError as e:
    typed_as(e)

try:
    raises(3)
except (KeyError, ValueError):
    typed_tuple(3)

try:
    raises(4)
except Exception:
    broad(4)

try:
    raises(5)
except:
    bare(5)

# The inner handler does not match, so the outer one runs.
try:
    try:
        raises(6)
    except KeyError:
        pass
except ValueError:
    nested_outer(6)

# The `try` body always raises, so its handler is the only way past the statement.
try:
    raises(7)
    raise AssertionError("unreached")
except ValueError:
    pass
after(7)

# A raise of a value rather than of a call.
error = ValueError(8)
try:
    raise error
except ValueError:
    raised_directly(8)


def guarded():
    try:
        return raises(9)
    except ValueError:
        return in_function(9)


guarded()

assert reached == [
    "typed",
    "typed_as",
    "typed_tuple",
    "broad",
    "bare",
    "nested_outer",
    "after",
    "raised_directly",
    "in_function",
], reached
