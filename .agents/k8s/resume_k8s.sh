#!/usr/bin/env bash
# Reconnect to a previously launched issue-775 TDM run.
#
# This script is meant to be run in a new session, possibly hours or
# days after the original launch. It is idempotent and safe to re-run.
#
# It will:
#   1. Verify the pod (recreating it from the manifest if the original
#      pod was evicted but the PVC still holds the staged repo/dataset).
#   2. Check the training done flag; if training is still running,
#      poll until it finishes (or until --timeout elapses).
#   3. Run the inference script (4-step student + 50-step teacher).
#   4. Pull the output tree from the pod to the workstation.
#   5. Leave the pod Running so the user can re-run inference with
#      different settings if needed.
#
# Usage:
#   KUBECONFIG=/home/toolbox/.kube/config \
#   RUN_NAME=tdm_bsz4_500_k8s_<timestamp> \
#       bash .agents/k8s/resume_k8s.sh
#
# Environment overrides:
#   KUBECONFIG, K8S_NAMESPACE, RUN_NAME, POD_NAME,
#   POLL_INTERVAL_SECONDS (default 120),
#   TIMEOUT_SECONDS (default 0 = wait forever for training),
#   SKIP_INFERENCE (default 0; set to 1 to only monitor + download),
#   SKIP_DOWNLOAD (default 0; set to 1 to only monitor + run inference),
#   RECREATE_POD (default 1; set to 0 to fail if the pod is gone).

set -euo pipefail

WORKSTATION_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${WORKSTATION_ROOT}"

KUBECONFIG="${KUBECONFIG:-/home/toolbox/.kube/config}"
export KUBECONFIG
K8S_NAMESPACE="${K8S_NAMESPACE:-vllm}"
: "${RUN_NAME:?ERROR: RUN_NAME must be set, e.g. RUN_NAME=tdm_bsz4_500_k8s_1719...}"
POD_NAME="${POD_NAME:-${RUN_NAME}}"
POLL_INTERVAL_SECONDS="${POLL_INTERVAL_SECONDS:-120}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-0}"
SKIP_INFERENCE="${SKIP_INFERENCE:-0}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
RECREATE_POD="${RECREATE_POD:-1}"
LOCAL_OUTPUT="${LOCAL_OUTPUT:-${WORKSTATION_ROOT}/outputs/issue-775-tdm/k8s/${RUN_NAME}}"
POD_OUTPUT_ROOT="/workspace/run/issue-775/${RUN_NAME}"

require_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "ERROR: required command '$1' not found on PATH" >&2
        exit 1
    fi
}
require_cmd kubectl
require_cmd python3
require_cmd tar

echo "=== K8s TDM resume (issue 775) ==="
echo "Namespace:        ${K8S_NAMESPACE}"
echo "Run name:         ${RUN_NAME}"
echo "Pod name:         ${POD_NAME}"
echo "Pod output root:  ${POD_OUTPUT_ROOT}"
echo "Local output:     ${LOCAL_OUTPUT}"
echo "Poll interval:    ${POLL_INTERVAL_SECONDS}s"
echo "Timeout:          ${TIMEOUT_SECONDS}s (0 = forever)"
echo "Skip inference:   ${SKIP_INFERENCE}"
echo "Skip download:    ${SKIP_DOWNLOAD}"
echo "Recreate pod:     ${RECREATE_POD}"
echo "=============================="

pod_exists() {
    kubectl --namespace "${K8S_NAMESPACE}" get pod "${POD_NAME}" \
        -o jsonpath='{.metadata.name}' 2>/dev/null
}

pod_phase() {
    kubectl --namespace "${K8S_NAMESPACE}" get pod "${POD_NAME}" \
        -o jsonpath='{.status.phase}' 2>/dev/null || echo "Missing"
}

POD_MANIFEST="${LOCAL_OUTPUT}/manifests/${POD_NAME}.yaml"

if [[ "$(pod_exists)" != "${POD_NAME}" ]]; then
    if [[ "${RECREATE_POD}" == "1" ]]; then
        if [[ ! -f "${POD_MANIFEST}" ]]; then
            echo "ERROR: pod ${POD_NAME} does not exist and no manifest at" >&2
            echo "       ${POD_MANIFEST}" >&2
            echo "       Re-run from a workstation that has the launch-time" >&2
            echo "       manifest (or set RUN_NAME to a value whose manifest" >&2
            echo "       is available)." >&2
            exit 1
        fi
        echo
        echo "=== Pod missing; recreating from ${POD_MANIFEST} ==="
        kubectl apply -f "${POD_MANIFEST}"
        kubectl --namespace "${K8S_NAMESPACE}" wait \
            --for=jsonpath='{.status.phase}'=Running \
            "pod/${POD_NAME}" --timeout=10m
        kubectl --namespace "${K8S_NAMESPACE}" wait \
            --for=jsonpath='{.status.initContainerStatuses[0].state.terminated.exitCode}'=0 \
            "pod/${POD_NAME}" --timeout=24h
    else
        echo "ERROR: pod ${POD_NAME} does not exist; set RECREATE_POD=1" >&2
        echo "       to recreate from the saved manifest." >&2
        exit 1
    fi
fi

echo
echo "=== Current pod phase: $(pod_phase) ==="

TRAIN_DONE_FLAG="${POD_OUTPUT_ROOT}/output/.train_done"
TRAIN_PID_FILE="${POD_OUTPUT_ROOT}/output/train.pid"

training_done() {
    kubectl --namespace "${K8S_NAMESPACE}" exec "${POD_NAME}" -- \
        test -f "${TRAIN_DONE_FLAG}" 2>/dev/null
}

training_process_alive() {
    local pid
    pid=$(kubectl --namespace "${K8S_NAMESPACE}" exec "${POD_NAME}" -- \
        cat "${TRAIN_PID_FILE}" 2>/dev/null || echo "")
    [[ -z "${pid}" ]] && return 1
    kubectl --namespace "${K8S_NAMESPACE}" exec "${POD_NAME}" -- \
        kill -0 "${pid}" 2>/dev/null
}

echo
echo "=== Waiting for training to complete ==="
start_ts=$(date +%s)
while true; do
    if training_done; then
        echo "    training done flag present"
        kubectl --namespace "${K8S_NAMESPACE}" exec "${POD_NAME}" -- \
            cat "${TRAIN_DONE_FLAG}" || true
        break
    fi
    if training_process_alive; then
        pid=$(kubectl --namespace "${K8S_NAMESPACE}" exec "${POD_NAME}" -- \
            cat "${TRAIN_PID_FILE}")
        last_row=$(kubectl --namespace "${K8S_NAMESPACE}" exec "${POD_NAME}" -- \
            tail -n 1 "${POD_OUTPUT_ROOT}/output/tracker/metrics.jsonl" 2>/dev/null \
            || true)
        elapsed=$(( $(date +%s) - start_ts ))
        echo "    t=${elapsed}s training pid=${pid} still alive"
        if [[ -n "${last_row}" ]]; then
            echo "      last tracker row:"
            echo "      ${last_row}" | head -c 200
            echo
        fi
    else
        echo "    training process is not alive and no done flag."
        echo "    last 50 lines of ${POD_OUTPUT_ROOT}/logs/train.log:"
        kubectl --namespace "${K8S_NAMESPACE}" exec "${POD_NAME}" -- \
            tail -n 50 "${POD_OUTPUT_ROOT}/logs/train.log" 2>/dev/null || true
        echo
        echo "    last 50 lines of ${POD_OUTPUT_ROOT}/logs/train.launcher.out:"
        kubectl --namespace "${K8S_NAMESPACE}" exec "${POD_NAME}" -- \
            tail -n 50 "${POD_OUTPUT_ROOT}/logs/train.launcher.out" 2>/dev/null || true
        exit 1
    fi
    if [[ "${TIMEOUT_SECONDS}" != "0" ]] && \
       (( $(date +%s) - start_ts > TIMEOUT_SECONDS )); then
        echo "    TIMEOUT_SECONDS=${TIMEOUT_SECONDS} reached; giving up wait"
        exit 1
    fi
    sleep "${POLL_INTERVAL_SECONDS}"
done

if [[ "${SKIP_INFERENCE}" == "1" ]]; then
    echo
    echo "=== SKIP_INFERENCE=1, skipping inference ==="
else
    echo
    echo "=== Running post-training inference ==="
    INFER_CMD="$(cat <<EOF
set -euo pipefail
cd '${POD_OUTPUT_ROOT}'
nohup bash repo/.agents/k8s/run_infer.sh '${POD_OUTPUT_ROOT}' \\
    > logs/infer.launcher.out 2>&1 < /dev/null &
INFER_PID=\$!
echo "\${INFER_PID}" > output/infer.pid
disown
wait "\${INFER_PID}"
EOF
)"
    kubectl --namespace "${K8S_NAMESPACE}" exec "${POD_NAME}" -- \
        bash -lc "${INFER_CMD}"
fi

if [[ "${SKIP_DOWNLOAD}" == "1" ]]; then
    echo
    echo "=== SKIP_DOWNLOAD=1, skipping artifact download ==="
else
    echo
    echo "=== Pulling artifacts from pod to workstation ==="
    mkdir -p "${LOCAL_OUTPUT}/output"
    kubectl --namespace "${K8S_NAMESPACE}" exec "${POD_NAME}" -- \
        tar -C "${POD_OUTPUT_ROOT}/output" -cf - . | \
        tar -C "${LOCAL_OUTPUT}/output" -xf -
    echo "    Copied pod output/ to ${LOCAL_OUTPUT}/output"

    mkdir -p "${LOCAL_OUTPUT}/validation"
    if kubectl --namespace "${K8S_NAMESPACE}" exec "${POD_NAME}" -- \
        test -d "${POD_OUTPUT_ROOT}/validation_step500" 2>/dev/null; then
        kubectl --namespace "${K8S_NAMESPACE}" exec "${POD_NAME}" -- \
            tar -C "${POD_OUTPUT_ROOT}" -cf - validation_step500 | \
            tar -C "${LOCAL_OUTPUT}/validation" -xf -
        echo "    Copied validation_step500 to ${LOCAL_OUTPUT}/validation"
    fi
fi

cat <<EOF

=================================================================
  Resume complete.
=================================================================
  Run name:        ${RUN_NAME}
  Pod name:        ${POD_NAME} (still Running; delete with:
                    kubectl -n ${K8S_NAMESPACE} delete pod ${POD_NAME})
  Local output:    ${LOCAL_OUTPUT}
  Pod output root: ${POD_OUTPUT_ROOT}

  Inspect locally:
    ls -lh ${LOCAL_OUTPUT}/output/checkpoint-*
    ls -lh ${LOCAL_OUTPUT}/output/tracker/metrics.jsonl
    ls -lh ${LOCAL_OUTPUT}/validation/validation_step500/student_tdm_4step/
    ls -lh ${LOCAL_OUTPUT}/validation/validation_step500/teacher_wan_50step/
EOF
