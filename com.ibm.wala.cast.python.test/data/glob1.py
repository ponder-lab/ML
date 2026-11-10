import glob
import os
import tempfile


def sink_name(n):
    pass


def sink_iname(n):
    pass


def sink_name1(n):
    pass


def sink_escaped(n):
    pass


def sink_joined(n):
    pass


directory = tempfile.mkdtemp()
with open(os.path.join(directory, "a.txt"), "w") as f:
    f.write("line\n")

for name in glob.glob(os.path.join(directory, "*.txt")):
    assert isinstance(name, str)
    sink_name(name)
    sink_joined("Error: File " + name)
for iname in glob.iglob(os.path.join(directory, "*.txt")):
    assert isinstance(iname, str)
    sink_iname(iname)
for name1 in glob.glob1(directory, "*.txt"):
    assert name1 == "a.txt"
    sink_name1(name1)
escaped = glob.escape(directory + "/[a]")
assert isinstance(escaped, str)
sink_escaped(escaped)
