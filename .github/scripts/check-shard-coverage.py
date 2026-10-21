#!/usr/bin/env python3
"""Check that the whole-project fixture shards together select every test in their category.

The build job leaves out the WholeProjectFixtures category, and each leg of the
whole-project-fixtures job selects its tests by an explicit -Dtest list. A test in the category
that no list matches therefore runs in neither job, and nothing reports it. This script lists every
test method in the category, tagged on its class or on the method itself, and fails if a shard list
matches none of them, naming each such test. It also fails on a shard pattern that matches no test
in the category, since a renamed test leaves its old pattern selecting nothing.

The shard lists are read from the workflow, as Surefire reads -Dtest: patterns separated by commas,
each a class name with an optional #method part, whose methods are separated by +, and * matching
any run of characters in either part.

Usage: check-shard-coverage.py [<workflow> [<test source root>]]
"""

import fnmatch
import os
import re
import sys

CATEGORY = "WholeProjectFixtures"
JOB = "whole-project-fixtures:"
WORKFLOW = ".github/workflows/continuous-integration.yml"
TEST_SOURCE = "com.ibm.wala.cast.python.ml.test/source"


def shard_patterns(workflow):
    """Return (shard, class pattern, method pattern) for each -Dtest pattern of the shard job."""
    with open(workflow) as f:
        lines = f.read().splitlines()
    start = next((i for i, l in enumerate(lines) if l.strip() == JOB), None)
    if start is None:
        sys.exit(f"No {JOB} job in {workflow}.")
    ret = []
    shard = None
    indent = len(lines[start]) - len(lines[start].lstrip())
    for line in lines[start + 1 :]:
        if line.strip() and len(line) - len(line.lstrip()) <= indent:
            break  # the next job
        m = re.match(r"\s*- shard:\s*(\S+)", line)
        if m:
            shard = m.group(1)
        m = re.match(r"\s*tests:\s*'([^']*)'", line)
        if m:
            for pattern in m.group(1).split(","):
                cls, _, methods = pattern.strip().partition("#")
                for method in methods.split("+") if methods else ["*"]:
                    ret.append((shard, cls, method))
    if not ret:
        sys.exit(f"No shard test lists found in the {JOB} job of {workflow}.")
    return ret


def category_tests(root):
    """Return (class, method) for each test method tagged with the category, on its class or itself."""
    ret = []
    for directory, _, files in os.walk(root):
        for name in files:
            if not name.endswith(".java"):
                continue
            with open(os.path.join(directory, name)) as f:
                text = f.read()
            if CATEGORY not in text:
                continue
            cls = name[: -len(".java")]
            tagged_class = False
            pending_test = pending_category = False
            for line in text.splitlines():
                stripped = line.strip()
                if stripped.startswith("@Category") and CATEGORY in stripped:
                    pending_category = True
                elif re.match(
                    r"(public\s+)?(abstract\s+)?(final\s+)?class\s+" + cls + r"\b",
                    stripped,
                ):
                    tagged_class = pending_category
                    pending_category = False
                elif stripped.startswith("@Test"):
                    pending_test = True
                else:
                    m = re.search(r"\bvoid\s+(\w+)\s*\(", stripped)
                    if m and pending_test:
                        if tagged_class or pending_category:
                            ret.append((cls, m.group(1)))
                        pending_test = pending_category = False
    return ret


def main():
    workflow = sys.argv[1] if len(sys.argv) > 1 else WORKFLOW
    root = sys.argv[2] if len(sys.argv) > 2 else TEST_SOURCE
    patterns = shard_patterns(workflow)
    tests = category_tests(root)
    if not tests:
        sys.exit(f"No test methods in the {CATEGORY} category under {root}.")

    def matches(pattern, test):
        return fnmatch.fnmatchcase(test[0], pattern[1]) and fnmatch.fnmatchcase(
            test[1], pattern[2]
        )

    unselected = [t for t in tests if not any(matches(p, t) for p in patterns)]
    unused = [p for p in patterns if not any(matches(p, t) for t in tests)]
    for cls, method in unselected:
        print(
            f"{cls}#{method} is in the {CATEGORY} category but in no shard's test list."
        )
    for shard, cls, method in unused:
        print(
            f"Shard {shard}'s pattern {cls}#{method} matches no test in the {CATEGORY} category."
        )
    if unselected or unused:
        sys.exit(1)
    print(
        f"All {len(tests)} tests in the {CATEGORY} category are in a shard's test list."
    )


if __name__ == "__main__":
    main()
