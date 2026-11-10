# Test for wala/ML#1242: a `+` of a file name from `glob.glob` and a line read from that file is a
# string concatenation, so the function the string is passed to receives no tensor.
import glob
import os
import tempfile


def file_lines_to_list(path):
    with open(path) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    return content


def error_joined(msg):
    assert isinstance(msg, str)
    print(msg)


def error_prefixed(msg):
    assert isinstance(msg, str)
    print(msg)


def error_chained(msg):
    assert isinstance(msg, str)
    print(msg)


def error_augmented(msg):
    assert isinstance(msg, str)
    print(msg)


def error_local_augmented(msg):
    assert isinstance(msg, str)
    print(msg)


def report_lines(name):
    for line in file_lines_to_list(name):
        message = "Error: File " + name + " in the wrong format.\n"
        message += " Received: " + line
        error_local_augmented(message)


directory = tempfile.mkdtemp()
with open(os.path.join(directory, "a.txt"), "w") as f:
    f.write("line\n")

for name in glob.glob(os.path.join(directory, "*.txt")):
    for line in file_lines_to_list(name):
        error_joined(name + line)
        error_prefixed(("Error: File " + name) + line)
        error_chained(
            "Error: File " + name + " in the wrong format.\n" + " Received: " + line
        )
        message = "Error: File " + name + " in the wrong format.\n"
        message += " Received: " + line
        error_augmented(message)
    report_lines(name)
