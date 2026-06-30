import argparse


def consume(value):
    pass


parser = argparse.ArgumentParser()
parser.add_argument("--data", nargs="*", default=[])
args = parser.parse_args([])

# `args.data` is an opaque (argparse) attribute, not a tensor. A subscript-slice of it is still not
# a tensor, so `value` should not be typed as a tensor. wala/ML#656.
sliced = args.data[1::2]
consume(sliced)
