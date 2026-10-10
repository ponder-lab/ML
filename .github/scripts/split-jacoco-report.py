#!/usr/bin/env python3
"""Split an aggregate JaCoCo report into one report per module source directory.

JaCoCo names a source file by its package and file name only. The jython and jython3 modules hold
files at the same package path, such as com/ibm/wala/cast/python/parser/PythonModuleParser.java, so
Codecov cannot tell the two copies apart and reports neither. Codecov first looks a file up as
<report name>/<package>/<file>, so this script writes each report group's files, one source
directory at a time, to a report named <module>/<source directory>, which makes that lookup the
file's path in the repository. Package and class names are left as they are, since Codecov keys
method complexity by class name. The aggregate report is replaced by the split reports.

Usage: split-jacoco-report.py <jacoco.xml> [<repository root>]
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
    aggregate = sys.argv[1]
    root = sys.argv[2] if len(sys.argv) > 2 else "."
    # ElementTree drops the report's DOCTYPE, which Codecov does not read.
    report = ET.parse(aggregate).getroot()
    sessions = report.findall("sessioninfo")
    split = {}
    for group in report.iter("group"):
        module = group.get("name")
        for package in group.findall("package"):
            name = package.get("name")
            by_root = {}
            for source in package.findall("sourcefile"):
                path = f"{name}/{source.get('name')}"
                by_root.setdefault(source_root(root, module, path), set()).add(
                    source.get("name")
                )
            for directory, files in by_root.items():
                copied = copy.deepcopy(package)
                for child in list(copied):
                    if child.tag == "sourcefile" and child.get("name") not in files:
                        copied.remove(child)
                    elif (
                        child.tag == "class"
                        and child.get("sourcefilename") not in files
                    ):
                        copied.remove(child)
                key = f"{module}/{directory}"
                if key not in split:
                    split[key] = ET.Element("report", {"name": key})
                    split[key].extend(copy.deepcopy(s) for s in sessions)
                split[key].append(copied)
    out = os.path.dirname(aggregate)
    count = 0
    for key, element in sorted(split.items()):
        target = os.path.join(out, "jacoco-" + key.replace("/", "-") + ".xml")
        ET.ElementTree(element).write(target, encoding="UTF-8", xml_declaration=True)
        count += len(element.findall("package/sourcefile"))
    os.remove(aggregate)
    print(
        f"Split {count} source files of {aggregate} into {len(split)} reports in {out}"
    )


if __name__ == "__main__":
    main()
