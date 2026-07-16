import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path

import modal

SOURCE_ROOT = Path("/src/FastVideo")
WORKSPACE_ROOT = Path("/workspace/FastVideo")


def _resolve_local_repo_root(module_path: Path, is_local: bool) -> Path:
    return module_path.resolve().parents[3] if is_local else SOURCE_ROOT


LOCAL_REPO_ROOT = _resolve_local_repo_root(Path(__file__), modal.is_local())
REQUIRED_SOURCE_DIRS = (
    Path("fastvideo-kernel/include/cutlass/include"),
    Path("fastvideo-kernel/include/tk/include"),
    Path("fastvideo/third_party/eval/vbench"),
)


def _run_git(repo_root: Path, *args: str) -> str:
    result = subprocess.run(["git", *args],
                            cwd=repo_root,
                            capture_output=True,
                            text=True,
                            check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed with exit code {result.returncode}: "
            f"{result.stderr.strip()}")
    return result.stdout.strip()


def _run_git_paths(repo_root: Path, *args: str) -> tuple[Path, ...]:
    result = subprocess.run(["git", *args],
                            cwd=repo_root,
                            capture_output=True,
                            check=False)
    if result.returncode != 0:
        stderr = os.fsdecode(result.stderr).strip()
        raise RuntimeError(
            f"git {' '.join(args)} failed with exit code {result.returncode}: "
            f"{stderr}")
    return tuple(
        Path(os.fsdecode(path)) for path in result.stdout.split(b"\0") if path)


def _validate_local_source_checkout(repo_root: Path,
                                    expected_revision: str | None) -> str:
    if expected_revision is not None and not expected_revision:
        raise RuntimeError(
            "BUILDKITE_COMMIT is required when running under Buildkite"
        )

    actual_commit = _run_git(repo_root, "rev-parse", "--verify",
                             "HEAD^{commit}")
    if expected_revision is not None:
        expected_commit = _run_git(repo_root, "rev-parse", "--verify",
                                   f"{expected_revision}^{{commit}}")
        if actual_commit.lower() != expected_commit.lower():
            raise RuntimeError(
                f"Buildkite checkout mismatch: HEAD is {actual_commit}, "
                f"expected {expected_commit}")

        status = _run_git(repo_root, "status", "--short",
                          "--untracked-files=all", "--", ".",
                          ":(exclude).agents/handoffs/**")
        if status:
            raise RuntimeError(
                "Buildkite checkout contains uncommitted source changes:\n" +
                status)

    submodule_status = _run_git(repo_root, "submodule", "status", "--recursive")
    invalid_submodules = [
        line for line in submodule_status.splitlines()
        if line and line[0] in "-+U"
    ]
    if invalid_submodules:
        raise RuntimeError(
            "Buildkite checkout has missing or mismatched submodules:\n" +
            "\n".join(invalid_submodules))

    return actual_commit


def _submodule_paths(submodule_status: str) -> tuple[Path, ...]:
    paths = []
    for line in submodule_status.splitlines():
        if not line:
            continue
        _, path_and_description = line[1:].split(maxsplit=1)
        paths.append(Path(path_and_description.rsplit(" (", maxsplit=1)[0]))
    return tuple(paths)


def _collect_local_source_paths(repo_root: Path) -> frozenset[Path]:
    paths = set(
        _run_git_paths(repo_root, "ls-files", "-z", "--cached", "--others",
                       "--exclude-standard"))
    submodule_status = _run_git(repo_root, "submodule", "status", "--recursive")
    for submodule_path in _submodule_paths(submodule_status):
        submodule_root = repo_root / submodule_path
        paths.update(submodule_path / path for path in _run_git_paths(
            submodule_root, "ls-files", "-z", "--cached", "--others",
            "--exclude-standard"))

    included_paths = {Path(".")}
    for path in paths:
        included_paths.add(path)
        included_paths.update(parent for parent in path.parents
                              if parent != Path("."))
    return frozenset(included_paths)


def _ignore_local_source(path: Path, *, repo_root: Path,
                         included_paths: frozenset[Path]) -> bool:
    try:
        relative_path = path.relative_to(repo_root) if path.is_absolute() else path
    except ValueError:
        return True

    parts = relative_path.parts
    if ".git" in parts or any(
            parts[index:index + 2] == (".agents", "handoffs")
            for index in range(len(parts) - 1)):
        return True
    return relative_path not in included_paths


def _build_local_source_ignore(repo_root: Path) -> Callable[[Path], bool]:
    return partial(_ignore_local_source,
                   repo_root=repo_root,
                   included_paths=_collect_local_source_paths(repo_root))


def _attach_local_source(image: modal.Image,
                         repo_root: Path = LOCAL_REPO_ROOT,
                         ignore: Callable[[Path], bool] | None = None) -> modal.Image:
    if ignore is None:
        ignore = _build_local_source_ignore(repo_root)
    return image.add_local_dir(repo_root,
                               remote_path=str(SOURCE_ROOT),
                               copy=False,
                               ignore=ignore)


def _prepare_workspace(source_root: Path = SOURCE_ROOT,
                       workspace_root: Path = WORKSPACE_ROOT) -> None:
    started_at = time.monotonic()
    if not (source_root / "pyproject.toml").is_file():
        raise RuntimeError(f"Attached FastVideo source is missing at {source_root}")

    missing_dirs = [
        str(relative_path) for relative_path in REQUIRED_SOURCE_DIRS
        if not (source_root / relative_path).is_dir()
        or not any((source_root / relative_path).iterdir())
    ]
    if missing_dirs:
        raise RuntimeError("Attached source is missing initialized submodules: " +
                           ", ".join(missing_dirs))

    shutil.rmtree(workspace_root, ignore_errors=True)
    workspace_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_root, workspace_root, symlinks=True)
    print(f"Prepared writable workspace at {workspace_root} from {source_root} "
          f"in {time.monotonic() - started_at:.2f}s",
          flush=True)


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from modal_image_utils import (  # noqa: E402
        resolve_image_ref, resolve_uv_torch_backend)
except ModuleNotFoundError:
    # Remote Modal containers re-import this module but mount only the
    # entrypoint file; the digest resolution already happened at local
    # launch time, so a passthrough is correct there.
    def resolve_image_ref(image_ref: str) -> str:
        return image_ref

    def resolve_uv_torch_backend(image_tag: str) -> str | None:
        return os.environ.get("UV_TORCH_BACKEND")

app = modal.App()

model_vol = modal.Volume.from_name("hf-model-weights")
image_version = os.getenv("IMAGE_VERSION", "latest")
image_tag = f"ghcr.io/hao-ai-lab/fastvideo/fastvideo-dev:{image_version}"
image_ref = resolve_image_ref(image_tag)
print(f"Using image: {image_ref}")

# Mutable tags inherit the registry image's baked backend, keeping a latest-tag
# transition safe. Explicit CUDA tags also work with older images that predate
# the baked setting, and a caller override always wins.
uv_torch_backend_override = resolve_uv_torch_backend(image_tag)

# INVARIANT: this image definition must be byte-identical for every CI job at
# a given base image digest -- one build, shared cache across all concurrent
# lanes. Never put a per-job/per-commit value (BUILDKITE_*, TEST_SCOPE,
# env-derived overrides) into the image via .env()/run_commands: it becomes an
# image layer, so whenever the base digest changes every concurrent job
# rebuilds its own image variant (~15-20 min each), blowing the Buildkite job
# budget. `image_ref` is the only env-derived input allowed here, because it
# *selects* the base digest. Per-job values arrive at runtime via
# `ci_env_secret` below.
base_image = (modal.Image.from_registry(
    image_ref, add_python="3.12"
).run_commands("rm -rf /FastVideo").apt_install(
    "cmake", "pkg-config", "build-essential", "curl", "libssl-dev", "ffmpeg"
).run_commands(
    "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable"
).run_commands("echo 'source ~/.cargo/env' >> ~/.bashrc").env({
    "PATH": "/root/.cargo/bin:$PATH",
    "HF_REPO_ID": "FastVideo/performance-tracking",
}))

dreamverse_base_image = (base_image.run_commands(
    "curl -fsSL https://deb.nodesource.com/setup_22.x | bash -"
).apt_install("nodejs").run_commands("node --version && npm --version"))

# Runtime attachments are deliberately added after every image build step.
# Source changes update the mount without rebuilding either shared base image.
buildkite_launch = os.environ.get("BUILDKITE") == "true"
source_commit = os.environ.get("BUILDKITE_COMMIT", "")
ci_commit = source_commit
if modal.is_local():
    expected_revision = source_commit if buildkite_launch else None
    source_commit = _validate_local_source_checkout(LOCAL_REPO_ROOT,
                                                    expected_revision)
    source_ignore = _build_local_source_ignore(LOCAL_REPO_ROOT)
    image = _attach_local_source(base_image, ignore=source_ignore)
    dreamverse_image = _attach_local_source(dreamverse_base_image,
                                            ignore=source_ignore)
    ci_commit = source_commit if buildkite_launch else ""
else:
    image = base_image
    dreamverse_image = dreamverse_base_image

# Per-job/per-invocation values are injected into the container environment at
# RUNTIME via this secret (attached to every function below), so the image
# stays identical across jobs. Consumers (workspace identity logging,
# fastvideo/tests/performance/{compare_baseline,identity}.py, the FA4
# resolver, `uv pip install`) all read os.environ at runtime, so nothing else
# changes.
ci_env_secret = modal.Secret.from_dict({
    "BUILDKITE_REPO": os.environ.get("BUILDKITE_REPO", ""),
    "BUILDKITE_COMMIT": ci_commit,
    "BUILDKITE_PULL_REQUEST": os.environ.get("BUILDKITE_PULL_REQUEST", ""),
    "BUILDKITE_BRANCH": os.environ.get("BUILDKITE_BRANCH", ""),
    "BUILDKITE_SOURCE": os.environ.get("BUILDKITE_SOURCE", ""),
    "BUILDKITE_BUILD_URL": os.environ.get("BUILDKITE_BUILD_URL", ""),
    "BUILDKITE_BUILD_ID": os.environ.get("BUILDKITE_BUILD_ID", ""),
    "BUILDKITE_JOB_ID": os.environ.get("BUILDKITE_JOB_ID", ""),
    "TEST_SCOPE": os.environ.get("TEST_SCOPE", ""),
    "IMAGE_VERSION": image_version,
    "FASTVIDEO_CONTAINER_IMAGE_REF": image_ref,
    "FASTVIDEO_SOURCE_COMMIT": source_commit,
    **{
        key: os.environ[key]
        for key in (
            "FASTVIDEO_ATTENTION_BACKEND",
            "FASTVIDEO_PERFORMANCE_PROFILE_VERSION",
        )
        if os.environ.get(key)
    },
    **({
        "UV_TORCH_BACKEND": uv_torch_backend_override
    } if uv_torch_backend_override else {}),
    # FA4 is opt-in (FASTVIDEO_FA4). Keep the default enabled for
    # inference/perf parity; model-load and training lanes that do not exercise
    # FA4 explicitly set FASTVIDEO_FA4=0 in their command strings below.
    # Caller override wins.
    "FASTVIDEO_FA4": os.environ.get("FASTVIDEO_FA4", "1"),
})

hf_secret = modal.Secret.from_dict(
    {"HF_API_KEY": os.environ.get("HF_API_KEY", "")})
wandb_secret = modal.Secret.from_dict(
    {"WANDB_API_KEY": os.environ.get("WANDB_API_KEY", "")})


def run_test(pytest_command: str):
    """Helper function to run a test suite with custom pytest command"""
    run_test_command(pytest_command, build_kernel=True)


def run_test_command(test_command: str,
                     build_kernel: bool,
                     install_command: str = 'uv pip install -e ".[test]"'):
    """Helper function to run a test suite with custom test command.

    Most FastVideo CI suites need the custom kernel build. App-level tests like
    DreamVerse's mock-backend UI checks do not, so keep the kernel build
    optional to avoid unrelated CUDA/kernel setup in that CI path.

    The dependency install runs BEFORE the kernel build: pyproject pins the
    PyPI fastvideo-kernel wheel, so an install after the build silently
    replaces the just-built in-tree kernel with the (older) wheel -- every
    lane would then test stale kernels. Pass install_command="" for commands
    that manage their own installs.
    """
    git_commit = os.environ.get("BUILDKITE_COMMIT", "")
    source_commit = os.environ.get("FASTVIDEO_SOURCE_COMMIT", "")
    if git_commit:
        print(f"Using attached Buildkite checkout at commit {git_commit}",
              flush=True)
    else:
        print(f"Using attached local source based on commit {source_commit}; "
              "the working tree may contain changes",
              flush=True)
    _prepare_workspace()

    build_kernel_command = """
    cd fastvideo-kernel &&
    ./build.sh &&
    cd .. &&
    """ if build_kernel else ""

    install_clause = f"{install_command} &&" if install_command else ""

    command = f"""
    source $HOME/.local/bin/env &&
    source /opt/venv/bin/activate &&
    cd {WORKSPACE_ROOT} &&
    {install_clause}
    {build_kernel_command}
    {test_command}
    """

    result = subprocess.run(["/bin/bash", "-c", command],
                            stdout=sys.stdout,
                            stderr=sys.stderr,
                            check=False)

    # Modal containers crash on sys.exit(0); raise on failure, return on success.
    if result.returncode != 0:
        raise RuntimeError(
            f"Test command failed with exit code {result.returncode}")

@app.function(gpu="H100:1",
              image=image,
              timeout=1200,
              secrets=[hf_secret, ci_env_secret],
              volumes={"/root/data": model_vol})
def run_encoder_tests():
    run_test(
        "export HF_HOME='/root/data/.cache' && hf auth login --token $HF_API_KEY && pytest ./fastvideo/tests/encoders -vs"
    )


@app.function(gpu="L40S:1",
              image=image,
              timeout=1200,
              secrets=[hf_secret, ci_env_secret],
              volumes={"/root/data": model_vol})
def run_vae_tests():
    run_test(
        "export HF_HOME='/root/data/.cache' && hf auth login --token $HF_API_KEY && pytest ./fastvideo/tests/vaes -vs"
    )


@app.function(gpu="L40S:1",
              image=image,
              timeout=900,
              secrets=[hf_secret, ci_env_secret],
              volumes={"/root/data": model_vol})
def run_transformer_tests():
    run_test(
        "export HF_HOME='/root/data/.cache' && hf auth login --token $HF_API_KEY && "
        "FASTVIDEO_FA4=0 pytest ./fastvideo/tests/transformers -vs"
    )


@app.function(gpu="L40S:4",
              cpu=8.0,
              memory=32768,
              image=image,
              timeout=900,
              secrets=[wandb_secret, ci_env_secret],
              volumes={"/root/data": model_vol})
def run_training_tests():
    run_test(
        "export HF_HOME='/root/data/.cache' && wandb login $WANDB_API_KEY && "
        "FASTVIDEO_FA4=0 pytest ./fastvideo/tests/training/Vanilla -srP"
    )


@app.function(gpu="L40S:2",
              cpu=8.0,
              memory=32768,
              image=image,
              timeout=900,
              secrets=[wandb_secret, ci_env_secret],
              volumes={"/root/data": model_vol})
def run_training_lora_tests():
    run_test(
        "export HF_HOME='/root/data/.cache' && wandb login $WANDB_API_KEY && "
        "FASTVIDEO_FA4=0 pytest ./fastvideo/tests/training/lora/test_lora_training.py -srP"
    )


@app.function(gpu="H100:2",
              image=image,
              timeout=900,
              secrets=[wandb_secret, ci_env_secret])
def run_training_tests_VSA():
    run_test(
        "wandb login $WANDB_API_KEY && FASTVIDEO_FA4=0 pytest ./fastvideo/tests/training/VSA -srP"
    )


@app.function(gpu="H100:1", image=image, timeout=900, secrets=[ci_env_secret])
def run_kernel_tests():
    run_test("pytest fastvideo-kernel/tests/ -vs")


# @app.function(gpu="H100:1", image=image, timeout=900, secrets=[ci_env_secret])
# def run_precision_tests_VSA():
#     # VSA correctness is covered by the same file now
#     run_test("pytest fastvideo-kernel/tests/test_correctness.py")

# @app.function(gpu="L40S:1", image=image, timeout=900, secrets=[ci_env_secret])
# def run_precision_tests_vmoba():
#     run_test("pytest fastvideo-kernel/tests/test_vmoba_correctness.py")


@app.function(gpu="L40S:1", image=image, timeout=900, secrets=[ci_env_secret])
def run_inference_tests_vmoba():
    run_test('python fastvideo/tests/inference/vmoba/test_vmoba_inference.py')


@app.function(gpu="L40S:1", image=image, timeout=1200, secrets=[ci_env_secret])
def run_inference_lora_tests():
    run_test(
        "pytest ./fastvideo/tests/inference/lora/test_lora_inference_similarity.py -vs"
    )


@app.function(gpu="L40S:2", image=image, timeout=900, secrets=[ci_env_secret])
def run_distill_dmd_tests():
    run_test(
        "FASTVIDEO_FA4=0 pytest ./fastvideo/tests/training/distill/test_distill_dmd.py -vs")


@app.function(gpu="L40S:2",
              image=image,
              timeout=900,
              secrets=[wandb_secret, ci_env_secret])
def run_self_forcing_tests():
    run_test(
        "wandb login $WANDB_API_KEY && "
        "FASTVIDEO_FA4=0 pytest ./fastvideo/tests/training/self-forcing/test_self_forcing.py -vs"
    )


@app.function(gpu="L40S:1", image=image, timeout=900, secrets=[ci_env_secret])
def run_unit_test():
    run_test(
        "pytest ./fastvideo/tests/api/ ./fastvideo/tests/contract/ ./fastvideo/tests/dataset/ "
        "./fastvideo/tests/workflow/ ./fastvideo/tests/entrypoints/ ./fastvideo/tests/train/ "
        "./fastvideo/tests/stages/ ./fastvideo/tests/ops/ ./fastvideo/tests/worker/ "
        "./fastvideo/tests/training/test_trackers.py "
        "./fastvideo/tests/attention/test_sdpa_metadata_mask_contract.py "
        "./fastvideo/tests/modal/test_pr_test.py "
        "--ignore=./fastvideo/tests/entrypoints/test_openai_api_integration.py "
        "--ignore=./fastvideo/tests/train/models --ignore=./fastvideo/tests/train/methods -vs"
    )


# TODO: David: GPU only used to resolve import time requirement (not needed for this test). Maybe make those imports lazy?
@app.function(gpu="L40S:1",
              image=dreamverse_image,
              timeout=1800,
              secrets=[ci_env_secret])
def run_dreamverse_app_tests():
    run_test_command(
        install_command="",
        build_kernel=False,
        test_command="""
        uv pip install -e ".[test,dreamverse]" &&
        export PYTHONPATH=/workspace/FastVideo/apps/dreamverse:$PYTHONPATH &&
        pytest apps/dreamverse/dreamverse/tests -q &&
        cd apps/dreamverse/web &&
        npm ci &&
        npm run typecheck &&
        npm test &&
        npx playwright install --with-deps chromium webkit firefox &&
        bash -c '
            set -e
            BACKEND_PORT="${BACKEND_PORT:-8009}"
            python -m uvicorn dreamverse.mock_server:app --host 127.0.0.1 --port "$BACKEND_PORT" &
            MOCK_SERVER_PID=$!
            trap "kill $MOCK_SERVER_PID 2>/dev/null || true" EXIT
            for i in {1..30}; do
                curl -fsS "http://127.0.0.1:$BACKEND_PORT/healthz" && break
                sleep 1
            done
            curl -fsS "http://127.0.0.1:$BACKEND_PORT/healthz"
            BACKEND_HOST=127.0.0.1 BACKEND_PORT="$BACKEND_PORT" CI=1 \
                npm run e2e -- \
                    --project=chromium \
                    --project=webkit \
                    --project=firefox \
                    --project=mobile-safari \
                    --project=mobile-chromium
        '
        """)


@app.function(gpu="L40S:1",
              cpu=8.0,
              memory=32768,
              image=image,
              timeout=1800,
              secrets=[hf_secret, ci_env_secret],
              volumes={"/root/data": model_vol})
def run_train_framework_tests():
    run_test(
        "export HF_HOME='/root/data/.cache' && hf auth login --token $HF_API_KEY && "
        "FASTVIDEO_FA4=0 pytest ./fastvideo/tests/train/models ./fastvideo/tests/train/methods -vs"
    )


@app.function(gpu="L40S:1",
              image=image,
              timeout=1800,
              secrets=[hf_secret, ci_env_secret],
              volumes={"/root/data": model_vol})
def seed_grad_norm_references():
    """Record the per-method grad-norm reference for the **CI GPU (L40S only)**.

    Phase 2 / 5a-ii one-off seeding entrypoint. Pinned to ``gpu="L40S:1"`` (the
    Modal CI runner), so this function only seeds the ``L40S`` key in
    ``fastvideo/tests/train/methods/grad_norm_refs.json``.

    ``FASTVIDEO_GRADNORM_UPDATE=1`` makes ``check_grad_norm_regression`` record
    the measured norm instead of asserting; ``-rs`` surfaces the recorded value
    in the log so it can be copied into the JSON.

    To seed any other device (e.g. our local Blackwell dev box → ``GB200``
    key), run the same env-var + pytest invocation directly on that
    workstation — see the module docstring of ``grad_norm_regression.py`` for
    the local command and the ``_DEVICE_MAPPINGS`` table.
    """
    run_test(
        "export HF_HOME='/root/data/.cache' && hf auth login --token $HF_API_KEY && "
        "FASTVIDEO_FA4=0 FASTVIDEO_GRADNORM_UPDATE=1 pytest ./fastvideo/tests/train/methods -vs -rs"
    )


@app.function(gpu="L40S:1",
              image=image,
              timeout=3600,
              secrets=[hf_secret, ci_env_secret],
              volumes={"/root/data": model_vol})
def run_eval_tests():
    # Eval metric regression: drives the high-level fastvideo.eval API on a
    # fixed asset and asserts each score matches the upstream reference number
    # checked into fastvideo/tests/eval/reference_scores/. Pulls several scorer
    # checkpoints (VideoScore2 VLM, VBench nets, audio models) on first run;
    # they cache on the hf-model-weights volume thereafter.
    #
    # Installs [eval-full] (eval + vbench + audio extras) on top of [test]:
    # the dev image only ships [dev], and without the extras skip_missing_deps
    # in conftest would silently drop nearly every metric and the lane would
    # pass vacuously. detectron2-backed vbench metrics remain skipped by
    # design (not pip-installable; see fastvideo/eval/README.md).
    run_test_command(
        "export HF_HOME='/root/data/.cache' && hf auth login --token $HF_API_KEY && pytest ./fastvideo/tests/eval -vs",
        build_kernel=True,
        install_command='uv pip install -e ".[test,eval-full]"')


@app.function(gpu="L40S:1",
              image=image,
              timeout=3600,
              secrets=[hf_secret, ci_env_secret])
def run_lora_extraction_tests():
    run_test(
        "hf auth login --token $HF_API_KEY && pytest ./fastvideo/tests/lora_extraction/test_lora_extraction.py"
    )


@app.function(gpu="L40S:2",
              cpu=8.0,
              memory=32768,
              image=image,
              timeout=1800,
              secrets=[hf_secret, ci_env_secret],
              volumes={"/root/data": model_vol})
def run_performance_tests():
    # PR/direct records are uploaded only on pass; scheduled main uploads pass
    # and fail so the dashboard records every canonical baseline attempt.
    run_test(
        "export HF_HOME='/root/data/.cache' && "
        "export PERFORMANCE_TRACKING_ROOT='/tmp/perf-tracking' && "
        "hf auth login --token $HF_API_KEY && "
        "if [ -n \"${BUILDKITE_PULL_REQUEST:-}\" ] && [ \"${BUILDKITE_PULL_REQUEST:-false}\" != 'false' ]; then "
        "export PERF_RUN_SOURCE='pr'; "
        "export PERF_UPLOAD_POLICY='pass'; "
        "elif [ \"${BUILDKITE_BRANCH:-}\" = 'main' ] && ( [ \"${BUILDKITE_SOURCE:-}\" = 'schedule' ] || [ \"${TEST_SCOPE:-}\" = 'full' ] ); then "
        "export PERF_RUN_SOURCE='scheduled_main'; "
        "export PERF_UPLOAD_POLICY='always'; "
        "elif [ \"${TEST_SCOPE:-}\" = 'direct' ]; then "
        "export PERF_RUN_SOURCE='unknown'; "
        "export PERF_UPLOAD_POLICY='pass'; "
        "else "
        "export PERF_RUN_SOURCE='unknown'; "
        "export PERF_UPLOAD_POLICY='never'; "
        "fi; "
        "(nvidia-smi --query-gpu=index,timestamp,clocks.sm,clocks.max.sm,power.draw,power.limit,temperature.gpu "
        "--format=csv -l 10 > /tmp/gpu_telemetry.csv 2>/dev/null &); "
        "pytest ./fastvideo/tests/performance -vs; "
        "PYTEST_RC=$?; "
        "PERF_RC=0; "
        "if [ $PYTEST_RC -eq 0 ] || [ \"$PERF_UPLOAD_POLICY\" = 'always' ]; then "
        "PERF_PYTEST_RC=$PYTEST_RC python ./fastvideo/tests/performance/compare_baseline.py; "
        "PERF_RC=$?; "
        "fi; "
        "python ./fastvideo/tests/performance/dashboard.py || true; "
        "echo '--- GPU telemetry (clocks.sm vs clocks.max.sm reveals capped hosts) ---'; "
        "cat /tmp/gpu_telemetry.csv || true; "
        "FINAL_RC=$PYTEST_RC; "
        "if [ $FINAL_RC -eq 0 ]; then FINAL_RC=$PERF_RC; fi; "
        "exit $FINAL_RC")


@app.function(gpu="L40S:1",
              image=image,
              timeout=1800,
              secrets=[hf_secret, ci_env_secret],
              volumes={"/root/data": model_vol})
def run_api_server_tests():
    run_test(
        "export HF_HOME='/root/data/.cache' && hf auth login --token $HF_API_KEY && pytest ./fastvideo/tests/entrypoints/test_openai_api_integration.py -vs"
    )
