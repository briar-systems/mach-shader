"""Lay a generated module out with `mach fmt -`, shared by the generators."""

import os
import re
import subprocess
import sys

# `mach fmt -` (stdin) first shipped in mach 5.1.0
MIN_MACH = (5, 1, 0)


def require_mach(tool, mach):
    try:
        info = subprocess.run([mach, "info"], capture_output=True, text=True).stdout
    except OSError as err:
        sys.exit(f"{tool}: cannot run the mach compiler '{mach}': {err}")
    found = re.match(r"mach (\d+)\.(\d+)\.(\d+)", info)
    want = ".".join(map(str, MIN_MACH))
    if not found:
        sys.exit(f"{tool}: could not read the version of '{mach}' from `mach info`; mach {want} or newer is required")
    have = tuple(int(n) for n in found.groups())
    if have < MIN_MACH:
        sys.exit(f"{tool}: '{mach}' is mach {'.'.join(map(str, have))}, but mach {want} or newer is required for `mach fmt -`; "
                 "install a newer compiler or point MACH at one")


def emit(tool, text):
    """Print `text` as `mach fmt` lays it out, with the compiler MACH names or the one on PATH."""
    mach = os.environ.get("MACH", "mach")
    require_mach(tool, mach)
    result = subprocess.run([mach, "fmt", "-"], input=text, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write("mach fmt rejected the generated file:\n" + result.stdout + result.stderr)
        sys.exit(1)
    print(result.stdout, end="")
