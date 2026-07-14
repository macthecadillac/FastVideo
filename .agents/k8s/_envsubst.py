#!/usr/bin/env python3
"""Minimal, whitelisted envsubst for the issue-775 K8s manifests."""

import os
import re
import sys

ALLOWED_SUBS = {
    "BRANCH",
    "COMMIT",
    "K8S_NAMESPACE",
    "POD_NAME",
    "POD_OUTPUT_ROOT",
    "PVC_NAME",
    "RUN_NAME",
    "SHARED_HF_HOME",
    "STAGE_POD_NAME",
    "SUPERVISOR_CONFIGMAP",
    "SUPERVISOR_JOB_NAME",
    "SUPERVISOR_RBAC_NAME",
    "SUPERVISOR_TIMEOUT_SECONDS",
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
