# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import stat
import subprocess
import sys
import types
from pathlib import Path

import pytest

COMMIT = "0123456789abcdef0123456789abcdef01234567"


class _FakeImage:

    def __init__(self, operations=()):
        self.operations = operations

    def _with(self, name, args, kwargs):
        return _FakeImage(self.operations + ((name, args, kwargs), ))

    @classmethod
    def from_registry(cls, *args, **kwargs):
        return cls((("from_registry", args, kwargs), ))

    def apt_install(self, *args, **kwargs):
        return self._with("apt_install", args, kwargs)

    def run_commands(self, *args, **kwargs):
        return self._with("run_commands", args, kwargs)

    def env(self, *args, **kwargs):
        return self._with("env", args, kwargs)

    def add_local_dir(self, *args, **kwargs):
        return self._with("add_local_dir", args, kwargs)


class _FakeVolume:

    @classmethod
    def from_name(cls, *_args, **_kwargs):
        return cls()


class _FakeSecret:

    def __init__(self, values):
        self.values = values

    @classmethod
    def from_dict(cls, values, **_kwargs):
        return cls(values)


class _FakeApp:

    def function(self, *_args, **_kwargs):

        def decorator(func):
            return func

        return decorator


def _load_pr_test_module(monkeypatch,
                         *,
                         is_local=False,
                         buildkite=False,
                         buildkite_commit=COMMIT):
    fake_modal = types.SimpleNamespace(
        App=lambda: _FakeApp(),
        Image=_FakeImage,
        Secret=_FakeSecret,
        Volume=_FakeVolume,
        is_local=lambda: is_local,
    )
    fake_image_utils = types.SimpleNamespace(
        resolve_image_ref=lambda image_ref: image_ref,
        resolve_uv_torch_backend=lambda _image_tag: None,
    )
    monkeypatch.setitem(sys.modules, "modal", fake_modal)
    monkeypatch.setitem(sys.modules, "modal_image_utils", fake_image_utils)

    monkeypatch.delenv("BUILDKITE", raising=False)
    monkeypatch.delenv("BUILDKITE_COMMIT", raising=False)
    if is_local:
        if buildkite:
            monkeypatch.setenv("BUILDKITE", "true")
            monkeypatch.setenv("BUILDKITE_COMMIT", buildkite_commit)

        def fake_run(command, **_kwargs):
            if command[:3] == ["git", "rev-parse", "--verify"]:
                revisions = {
                    "HEAD^{commit}",
                    f"{buildkite_commit}^{{commit}}",
                }
                if command[3] not in revisions:
                    raise AssertionError(f"Unexpected revision: {command}")
                stdout = COMMIT
            elif command[:2] == ["git", "status"]:
                stdout = ""
            elif command[:3] == ["git", "submodule", "status"]:
                stdout = (
                    " e67e63c331d6 fastvideo-kernel/include/cutlass\n"
                    " 6c27e28c8115 fastvideo-kernel/include/tk\n"
                    " 45e79ec14e69 fastvideo/third_party/eval/vbench")
            else:
                raise AssertionError(f"Unexpected command: {command}")
            return types.SimpleNamespace(returncode=0, stdout=stdout, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)

    module_path = Path(__file__).with_name("pr_test.py")
    spec = importlib.util.spec_from_file_location("modal_pr_test_under_test",
                                                  module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_manual_runtime_source_attachment_needs_no_buildkite_metadata(
        monkeypatch):
    module = _load_pr_test_module(monkeypatch, is_local=True)

    assert module.ci_env_secret.values["BUILDKITE_COMMIT"] == COMMIT

    for attached_image in (module.image, module.dreamverse_image):
        assert attached_image.operations[-1][0] == "add_local_dir"


def test_runtime_source_attachment_is_last_image_operation(monkeypatch):
    module = _load_pr_test_module(monkeypatch,
                                  is_local=True,
                                  buildkite=True)

    for attached_image in (module.image, module.dreamverse_image):
        operation, args, kwargs = attached_image.operations[-1]
        assert operation == "add_local_dir"
        assert Path(args[0]) == module.LOCAL_REPO_ROOT
        assert kwargs["remote_path"] == "/src/FastVideo"
        assert kwargs["copy"] is False
        assert kwargs["ignore"] is module._ignore_local_source
        assert all(
            name != "add_local_dir"
            for name, _, _ in attached_image.operations[:-1])

    assert any(
        name == "apt_install" and "nodejs" in args
        for name, args, _ in module.dreamverse_image.operations)


def test_buildkite_symbolic_revision_resolves_to_commit(monkeypatch):
    module = _load_pr_test_module(monkeypatch,
                                  is_local=True,
                                  buildkite=True,
                                  buildkite_commit="HEAD")

    assert module.ci_env_secret.values["BUILDKITE_COMMIT"] == COMMIT


def test_source_ignore_excludes_git_and_handoff_state(monkeypatch):
    module = _load_pr_test_module(monkeypatch)

    assert module._ignore_local_source(Path(".git/config"))
    assert module._ignore_local_source(Path("fastvideo-kernel/include/tk/.git"))
    assert module._ignore_local_source(
        Path(".agents/handoffs/issue-1611-handoff.md"))
    assert not module._ignore_local_source(Path(".agents/skills/add-model/SKILL.md"))
    assert not module._ignore_local_source(Path("fastvideo/__init__.py"))


def test_local_source_validation_accepts_exact_clean_checkout(monkeypatch):
    module = _load_pr_test_module(monkeypatch)
    calls = []

    def fake_git(_repo_root, *args):
        calls.append(args)
        if args == ("rev-parse", "--verify", "HEAD^{commit}"):
            return COMMIT.upper()
        if args == ("rev-parse", "--verify", f"{COMMIT}^{{commit}}"):
            return COMMIT
        if args[0] == "status":
            return ""
        if args[:2] == ("submodule", "status"):
            return " abc path/one\n def path/two"
        raise AssertionError(args)

    monkeypatch.setattr(module, "_run_git", fake_git)
    actual_commit = module._validate_local_source_checkout(
        Path("/repo"), COMMIT)

    assert actual_commit == COMMIT.upper()
    assert calls[0] == ("rev-parse", "--verify", "HEAD^{commit}")
    assert calls[1] == ("rev-parse", "--verify", f"{COMMIT}^{{commit}}")
    assert ":(exclude).agents/handoffs/**" in calls[2]
    assert calls[3] == ("submodule", "status", "--recursive")


def test_manual_source_validation_allows_working_tree_changes(monkeypatch):
    module = _load_pr_test_module(monkeypatch)

    def fake_git(_repo_root, *args):
        if args == ("rev-parse", "--verify", "HEAD^{commit}"):
            return COMMIT
        if args[:2] == ("submodule", "status"):
            return " abc path"
        raise AssertionError(args)

    monkeypatch.setattr(module, "_run_git", fake_git)

    assert module._validate_local_source_checkout(Path("/repo"),
                                                  None) == COMMIT


@pytest.mark.parametrize(
    ("actual_commit", "status", "submodules", "error"),
    [
        ("f" * 40, "", " abc path", "checkout mismatch"),
        (COMMIT, " M fastvideo/tests/modal/pr_test.py", " abc path",
         "uncommitted source changes"),
        (COMMIT, "", "-abc path", "missing or mismatched submodules"),
        (COMMIT, "", "+abc path", "missing or mismatched submodules"),
    ],
)
def test_local_source_validation_rejects_invalid_checkout(
        monkeypatch, actual_commit, status, submodules, error):
    module = _load_pr_test_module(monkeypatch)

    def fake_git(_repo_root, *args):
        if args == ("rev-parse", "--verify", "HEAD^{commit}"):
            return actual_commit
        if args == ("rev-parse", "--verify", f"{COMMIT}^{{commit}}"):
            return COMMIT
        if args[0] == "status":
            return status
        if args[:2] == ("submodule", "status"):
            return submodules
        raise AssertionError(args)

    monkeypatch.setattr(module, "_run_git", fake_git)
    with pytest.raises(RuntimeError, match=error):
        module._validate_local_source_checkout(Path("/repo"), COMMIT)


def test_buildkite_source_validation_requires_revision(monkeypatch):
    module = _load_pr_test_module(monkeypatch)
    monkeypatch.setattr(
        module, "_run_git",
        lambda *_args: pytest.fail("Git should not run without a revision"))

    with pytest.raises(RuntimeError, match="BUILDKITE_COMMIT is required"):
        module._validate_local_source_checkout(Path("/repo"), "")


def _make_source_tree(module, root):
    (root / "pyproject.toml").write_text("[project]\nname='test'\n",
                                         encoding="utf-8")
    for relative_path in module.REQUIRED_SOURCE_DIRS:
        submodule_dir = root / relative_path
        submodule_dir.mkdir(parents=True)
        (submodule_dir / "sentinel").write_text("present", encoding="utf-8")

    script = root / "tool.sh"
    script.write_text("#!/bin/sh\necho ok\n", encoding="utf-8")
    script.chmod(0o755)
    (root / "tool-link").symlink_to("tool.sh")


def test_prepare_workspace_creates_isolated_writable_copies(monkeypatch,
                                                            tmp_path):
    module = _load_pr_test_module(monkeypatch)
    source = tmp_path / "source"
    workspace_a = tmp_path / "container-a" / "FastVideo"
    workspace_b = tmp_path / "container-b" / "FastVideo"
    source.mkdir()
    _make_source_tree(module, source)

    workspace_a.mkdir(parents=True)
    (workspace_a / "stale-output").write_text("old", encoding="utf-8")

    module._prepare_workspace(source, workspace_a)
    module._prepare_workspace(source, workspace_b)

    assert not (workspace_a / "stale-output").exists()
    assert (workspace_a / "tool-link").is_symlink()
    assert (workspace_a / "tool.sh").stat().st_mode & stat.S_IXUSR

    (workspace_a / "tool.sh").write_text("changed", encoding="utf-8")
    assert (source / "tool.sh").read_text(encoding="utf-8").startswith("#!")
    assert (workspace_b / "tool.sh").read_text(encoding="utf-8").startswith("#!")


def test_prepare_workspace_requires_initialized_submodules(monkeypatch,
                                                           tmp_path):
    module = _load_pr_test_module(monkeypatch)
    source = tmp_path / "source"
    source.mkdir()
    (source / "pyproject.toml").write_text("[project]\nname='test'\n",
                                           encoding="utf-8")

    with pytest.raises(RuntimeError, match="missing initialized submodules"):
        module._prepare_workspace(source, tmp_path / "workspace")


@pytest.mark.parametrize(
    ("build_kernel", "install_command"),
    [
        (True, 'uv pip install -e ".[test]"'),
        (False, ""),
    ],
)
def test_run_test_command_uses_writable_workspace_without_git(
        monkeypatch, build_kernel, install_command):
    module = _load_pr_test_module(monkeypatch)
    real_run = subprocess.run
    events = []
    monkeypatch.setenv("BUILDKITE_COMMIT", COMMIT)
    monkeypatch.setattr(
        module,
        "_prepare_workspace",
        lambda: events.append(("prepare", None)),
    )

    def fake_run(args, **_kwargs):
        events.append(("run", args))
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    module.run_test_command("pytest fastvideo/tests/api -q",
                            build_kernel=build_kernel,
                            install_command=install_command)

    assert [name for name, _ in events] == ["prepare", "run"]
    shell_command = events[1][1][-1]
    assert "cd /workspace/FastVideo" in shell_command
    assert "git clone" not in shell_command
    assert "git fetch" not in shell_command
    assert "git checkout" not in shell_command
    assert "git submodule" not in shell_command
    assert ("./build.sh" in shell_command) is build_kernel
    if install_command:
        assert install_command in shell_command
    else:
        assert "uv pip install" not in shell_command
    real_run(["/bin/bash", "-n"], input=shell_command, text=True, check=True)


def test_ci_collects_workspace_regression_and_dreamverse_uses_workspace(
        monkeypatch):
    module = _load_pr_test_module(monkeypatch)
    source = Path(module.__file__).read_text(encoding="utf-8")

    assert "./fastvideo/tests/modal/test_pr_test.py" in source
    assert "PYTHONPATH=/workspace/FastVideo/apps/dreamverse" in source
    assert "PYTHONPATH=/FastVideo/apps/dreamverse" not in source
