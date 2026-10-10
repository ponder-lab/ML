#!/usr/bin/env python3
"""Rewrite an aggregate JaCoCo report's package names to repository paths.

JaCoCo names a source file by its package and file name only. The jython and jython3 modules hold
files at the same package path, such as com/ibm/wala/cast/python/parser/PythonModuleParser.java, so
Codecov cannot tell the two copies apart and reports neither. Each report group is a module, so this
script renames every package to <module>/<source directory>/<package>, the file's path in the
repository, splitting a package whose files lie under more than one source directory.

Usage: qualify-jacoco-paths.py <jacoco.xml> [<repository root>]
"""

import copy
import os
import sys
import xml.etree.ElementTree as ET

SOURCE_DIRECTORIES = ("source", "test-source", "src/main/java", "src/test/java")


def source_root(root, module, path):
    """Return the module's single source directory holding path, or exit if there is not one."""
    hits = [
        d
        for d in SOURCE_DIRECTORIES
        if os.path.isfile(os.path.join(root, module, d, path))
    ]
    if len(hits) != 1:
        sys.exit(
            f"{module}: {path} is under {hits or 'no'} source directory, expected one"
        )
    return hits[0]


def main():
    report = sys.argv[1]
    root = sys.argv[2] if len(sys.argv) > 2 else "."
    # ElementTree drops the report's DOCTYPE, which Codecov does not read.
    tree = ET.parse(report)
    renamed = 0
    for group in tree.getroot().iter("group"):
        module = group.get("name")
        for package in list(group.findall("package")):
            name = package.get("name")
            by_root = {}
            for source in package.findall("sourcefile"):
                path = f"{name}/{source.get('name')}"
                by_root.setdefault(source_root(root, module, path), []).append(source)
            if not by_root:
                continue
            index = list(group).index(package)
            group.remove(package)
            for directory, sources in sorted(by_root.items()):
                files = {s.get("name") for s in sources}
                split = copy.deepcopy(package)
                split.set("name", f"{module}/{directory}/{name}")
                for child in list(split):
                    if child.tag == "sourcefile" and child.get("name") not in files:
                        split.remove(child)
                    elif (
                        child.tag == "class"
                        and child.get("sourcefilename") not in files
                    ):
                        split.remove(child)
                group.insert(index, split)
                index += 1
                renamed += len(sources)
    tree.write(report, encoding="UTF-8", xml_declaration=True)
    print(f"Qualified {renamed} source files in {report}")


if __name__ == "__main__":
    main()
