# Test https://github.com/wala/ML/issues/977: this script lies OUTSIDE every PYTHONPATH entry (the
# path is `src` only) and contains an import. The translation must fail with a message that names
# this script and the path, not with a NullPointerException.
from pkg.a import f

print(f(1))
