#!/usr/bin/env python3
"""Minimal envsubst for the issue-775 K8s pod manifest.

Substitutes only `${VAR}` references whose names appear in the
ALLOWED_SUBS whitelist. Any other `${VAR}` (e.g. `${GH_TOKEN}` and
`${HF_TOKEN}`) is left intact for the container's shell to expand at
runtime, when the env has those values.

Usage:
    VAR=value VAR2=value2 python3 _envsubst.py < in.yaml > out.yaml
"""

import os
import re
import sys

ALLOWED_SUBS = {
    "POD_NAME",
    "POD_OUTPUT_ROOT",
    "K8S_NAMESPACE",
    "BRANCH",
    "COMMIT",
    "RUN_NAME",
}

src = sys.stdin.read()


def repl(match):
    name = match.group(1)
    if name not in ALLOWED_SUBS:
        return match.group(0)
    if name not in os.environ:
        raise SystemExit(f"envsubst: variable {name!r} is not set")
    return os.environ[name]


out = re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", repl, src)
sys.stdout.write(out)
