#!/usr/bin/env bash
# Reconnect, optionally recover from the latest checkpoint, validate outputs,
# run comparisons, and download artifacts for an issue-775 K8s run.

set -euo pipefail

WORKSTATION_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${WORKSTATION_ROOT}"
KUBECONFIG="${KUBECONFIG:-/home/sandbox/.kube/config}"
export KUBECONFIG
K8S_NAMESPACE="${K8S_NAMESPACE:-vllm}"
: "${RUN_NAME:?ERROR: RUN_NAME must be set}"
POD_NAME="${POD_NAME:-${RUN_NAME}}"
POLL_INTERVAL_SECONDS="${POLL_INTERVAL_SECONDS:-120}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-0}"
SKIP_INFERENCE="${SKIP_INFERENCE:-0}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
RERUN_INFERENCE="${RERUN_INFERENCE:-0}"
RECREATE_POD="${RECREATE_POD:-1}"
LOCAL_OUTPUT="${LOCAL_OUTPUT:-${WORKSTATION_ROOT}/outputs/issue-775-tdm/k8s/${RUN_NAME}}"
POD_OUTPUT_ROOT="/workspace/run/issue-775/${RUN_NAME}"
POD_MANIFEST="${LOCAL_OUTPUT}/manifests/${POD_NAME}.yaml"
POD_RECREATED=0
mkdir -p "${LOCAL_OUTPUT}"

for command in kubectl python3 tar; do
    command -v "${command}" >/dev/null 2>&1 || {
        echo "ERROR: required command '${command}' not found" >&2
        exit 1
    }
done

pod_exists() {
    kubectl --namespace "${K8S_NAMESPACE}" get pod "${POD_NAME}" \
        -o jsonpath='{.metadata.name}' 2>/dev/null
}
if [[ "$(pod_exists)" != "${POD_NAME}" ]]; then
    if [[ "${RECREATE_POD}" != "1" || ! -f "${POD_MANIFEST}" ]]; then
        echo "ERROR: pod is missing and cannot be recreated from ${POD_MANIFEST}" >&2
        exit 1
    fi
    echo "=== Recreating GPU pod from durable manifest ==="
    kubectl apply --dry-run=server -f "${POD_MANIFEST}" >/dev/null
    kubectl apply -f "${POD_MANIFEST}"
    kubectl --namespace "${K8S_NAMESPACE}" wait --for=condition=Ready \
        "pod/${POD_NAME}" --timeout=15m
    POD_RECREATED=1
fi

kubectl --namespace "${K8S_NAMESPACE}" exec "${POD_NAME}" -- \
    test -f "${POD_OUTPUT_ROOT}/.stage_done"
kubectl --namespace "${K8S_NAMESPACE}" exec "${POD_NAME}" -- \
    test -d "${POD_OUTPUT_ROOT}/repo/.git"

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
    [[ -n "${pid}" ]] && kubectl --namespace "${K8S_NAMESPACE}" exec "${POD_NAME}" -- \
        kill -0 "${pid}" 2>/dev/null
}
launch_from_latest() {
    local command
    command="$(cat <<EOF
set -euo pipefail
cd '${POD_OUTPUT_ROOT}'
nohup env RESUME_FROM_CHECKPOINT=latest \\
    bash repo/.agents/k8s/run_train.sh '${POD_OUTPUT_ROOT}' \\
    > logs/train.launcher.out 2>&1 < /dev/null &
TRAIN_PID=\$!
echo \"\${TRAIN_PID}\" > output/train.pid
disown
sleep 2
kill -0 \"\${TRAIN_PID}\"
EOF
)"
    kubectl --namespace "${K8S_NAMESPACE}" exec "${POD_NAME}" -- bash -lc "${command}"
}

if (( POD_RECREATED == 1 )) && ! training_done; then
    if kubectl --namespace "${K8S_NAMESPACE}" exec "${POD_NAME}" -- \
        sh -c "ls '${POD_OUTPUT_ROOT}'/output/checkpoint-*/metadata.json >/dev/null 2>&1"; then
        echo "=== Pod was recreated; resuming training from latest checkpoint ==="
        launch_from_latest
    else
        echo "ERROR: pod was lost before the first durable checkpoint" >&2
        exit 1
    fi
fi

echo "=== Waiting for training completion ==="
start_ts=$(date +%s)
while true; do
    if training_done; then
        status=$(kubectl --namespace "${K8S_NAMESPACE}" exec "${POD_NAME}" -- \
            cat "${TRAIN_DONE_FLAG}")
        echo "${status}"
        if [[ "${status}" != *"rc=0"* ]]; then
            echo "ERROR: training finished unsuccessfully" >&2
            kubectl --namespace "${K8S_NAMESPACE}" exec "${POD_NAME}" -- \
                tail -n 100 "${POD_OUTPUT_ROOT}/logs/train.log" || true
            exit 1
        fi
        break
    fi
    if ! training_process_alive; then
        echo "ERROR: training process is not alive and no completion status exists" >&2
        kubectl --namespace "${K8S_NAMESPACE}" exec "${POD_NAME}" -- \
            tail -n 100 "${POD_OUTPUT_ROOT}/logs/train.launcher.out" || true
        exit 1
    fi
    pid=$(kubectl --namespace "${K8S_NAMESPACE}" exec "${POD_NAME}" -- \
        cat "${TRAIN_PID_FILE}")
    echo "    training pid=${pid} still alive"
    kubectl --namespace "${K8S_NAMESPACE}" exec "${POD_NAME}" -- \
        tail -n 1 "${POD_OUTPUT_ROOT}/output/tracker/metrics.jsonl" 2>/dev/null | head -c 300 || true
    echo
    if [[ "${TIMEOUT_SECONDS}" != "0" ]] && \
       (( $(date +%s) - start_ts > TIMEOUT_SECONDS )); then
        echo "ERROR: timeout waiting for training" >&2
        exit 1
    fi
    sleep "${POLL_INTERVAL_SECONDS}"
done

kubectl --namespace "${K8S_NAMESPACE}" exec "${POD_NAME}" -- \
    python "${POD_OUTPUT_ROOT}/repo/.agents/k8s/validate_output.py" \
        "${POD_OUTPUT_ROOT}/output"

INFER_DONE="${POD_OUTPUT_ROOT}/validation_step500/.infer_done"
if [[ "${SKIP_INFERENCE}" == "1" ]]; then
    echo "=== Inference skipped ==="
elif [[ "${RERUN_INFERENCE}" != "1" ]] && \
     kubectl --namespace "${K8S_NAMESPACE}" exec "${POD_NAME}" -- test -f "${INFER_DONE}"; then
    echo "=== Inference already complete; set RERUN_INFERENCE=1 to repeat ==="
else
    echo "=== Running strict post-training comparisons ==="
    kubectl --namespace "${K8S_NAMESPACE}" exec "${POD_NAME}" -- \
        bash "${POD_OUTPUT_ROOT}/repo/.agents/k8s/run_infer.sh" "${POD_OUTPUT_ROOT}" \
        2>&1 | tee "${LOCAL_OUTPUT}/inference.stream.log"
fi

if [[ "${SKIP_DOWNLOAD}" != "1" ]]; then
    mkdir -p "${LOCAL_OUTPUT}/output" "${LOCAL_OUTPUT}/validation"
    kubectl --namespace "${K8S_NAMESPACE}" exec "${POD_NAME}" -- \
        tar -C "${POD_OUTPUT_ROOT}/output" -cf - . | \
        tar -C "${LOCAL_OUTPUT}/output" -xf -
    if kubectl --namespace "${K8S_NAMESPACE}" exec "${POD_NAME}" -- \
        test -d "${POD_OUTPUT_ROOT}/validation_step500"; then
        kubectl --namespace "${K8S_NAMESPACE}" exec "${POD_NAME}" -- \
            tar -C "${POD_OUTPUT_ROOT}" -cf - validation_step500 | \
            tar -C "${LOCAL_OUTPUT}/validation" -xf -
    fi
    kubectl --namespace "${K8S_NAMESPACE}" exec "${POD_NAME}" -- \
        tar -C "${POD_OUTPUT_ROOT}" -cf - logs preflight .stage_done \
        .model_snapshot .dataset_snapshot | \
        tar -C "${LOCAL_OUTPUT}" -xf -
fi

cat <<EOF
Resume complete.
Run name:     ${RUN_NAME}
Pod:          ${POD_NAME} (left Running for verification)
Local output: ${LOCAL_OUTPUT}
Delete after verification:
  kubectl --kubeconfig ${KUBECONFIG} -n ${K8S_NAMESPACE} delete pod ${POD_NAME}
EOF
