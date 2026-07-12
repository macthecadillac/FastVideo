#!/usr/bin/env bash
# Stage data without GPUs, run a four-GB200 preflight, then launch training.
# This script performs cluster mutations when invoked; do not run it until the
# user separately approves the launch.

set -euo pipefail

WORKSTATION_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${WORKSTATION_ROOT}"

KUBECONFIG="${KUBECONFIG:-/home/sandbox/.kube/config}"
export KUBECONFIG
K8S_NAMESPACE="${K8S_NAMESPACE:-vllm}"
BRANCH="${BRANCH:-issue/775-tdm}"
COMMIT="${COMMIT:-$(git rev-parse "origin/${BRANCH}")}"
RUN_NAME="${RUN_NAME:-tdm-bsz4-500-k8s-$(date +%s)}"
POD_NAME="${POD_NAME:-${RUN_NAME}}"
STAGE_POD_NAME="${STAGE_POD_NAME:-${POD_NAME}-stage}"
PVC_NAME="${PVC_NAME:-lustre-pvc-vllm}"
POLL_SECONDS="${POLL_SECONDS:-90}"
STAGE_TIMEOUT_SECONDS="${STAGE_TIMEOUT_SECONDS:-86400}"
LOCAL_OUTPUT="${LOCAL_OUTPUT:-${WORKSTATION_ROOT}/outputs/issue-775-tdm/k8s/${RUN_NAME}}"
POD_OUTPUT_ROOT="/workspace/run/issue-775/${RUN_NAME}"
SHARED_HF_HOME="/workspace/run/issue-775/shared-cache/hf"

for command in kubectl git python3 tar; do
    command -v "${command}" >/dev/null 2>&1 || {
        echo "ERROR: required command '${command}' not found" >&2
        exit 1
    }
done
if (( ${#POD_NAME} > 63 || ${#STAGE_POD_NAME} > 63 )); then
    echo "ERROR: pod names must be at most 63 characters" >&2
    exit 1
fi
kubectl --namespace "${K8S_NAMESPACE}" get nodes -o name >/dev/null

mkdir -p "${LOCAL_OUTPUT}/manifests"
POD_MANIFEST="${LOCAL_OUTPUT}/manifests/${POD_NAME}.yaml"
STAGE_MANIFEST="${LOCAL_OUTPUT}/manifests/${STAGE_POD_NAME}.yaml"
render_manifest() {
    local source=$1
    local destination=$2
    POD_OUTPUT_ROOT="${POD_OUTPUT_ROOT}" \
    POD_NAME="${POD_NAME}" \
    STAGE_POD_NAME="${STAGE_POD_NAME}" \
    K8S_NAMESPACE="${K8S_NAMESPACE}" \
    BRANCH="${BRANCH}" \
    COMMIT="${COMMIT}" \
    RUN_NAME="${RUN_NAME}" \
    PVC_NAME="${PVC_NAME}" \
    SHARED_HF_HOME="${SHARED_HF_HOME}" \
        python3 .agents/k8s/_envsubst.py < "${source}" > "${destination}"
}
render_manifest .agents/k8s/stage-pod.yaml "${STAGE_MANIFEST}"
render_manifest .agents/k8s/pod.yaml "${POD_MANIFEST}"

cat <<EOF
=== K8s TDM launch (issue 775) ===
Namespace:        ${K8S_NAMESPACE}
Cluster:          $(kubectl config current-context)
Run name:         ${RUN_NAME}
Stage pod:        ${STAGE_POD_NAME}
GPU pod:          ${POD_NAME}
Commit:           ${COMMIT}
PVC:              ${PVC_NAME}
Shared HF cache:  ${SHARED_HF_HOME}
Pod output root:  ${POD_OUTPUT_ROOT}
Local output:     ${LOCAL_OUTPUT}
==================================
EOF

kubectl apply --dry-run=server -f "${STAGE_MANIFEST}" >/dev/null
kubectl apply --dry-run=server -f "${POD_MANIFEST}" >/dev/null
echo "Server-side dry-run accepted both manifests."

for name in "${STAGE_POD_NAME}" "${POD_NAME}"; do
    if kubectl --namespace "${K8S_NAMESPACE}" get pod "${name}" -o name >/dev/null 2>&1; then
        echo "ERROR: pod ${name} already exists; choose a new RUN_NAME" >&2
        exit 1
    fi
done

echo "=== Applying CPU-only staging pod ==="
kubectl apply -f "${STAGE_MANIFEST}"

poll_start=$(date +%s)
poll_deadline=$(( poll_start + POLL_SECONDS ))
samples=()
while (( $(date +%s) < poll_deadline && ${#samples[@]} < 5 )); do
    phase=$(kubectl --namespace "${K8S_NAMESPACE}" get pod "${STAGE_POD_NAME}" \
        -o jsonpath='{.status.phase}' 2>/dev/null || echo Missing)
    if [[ "${phase}" == "Failed" ]]; then
        kubectl --namespace "${K8S_NAMESPACE}" logs "${STAGE_POD_NAME}" || true
        exit 1
    fi
    if [[ "${phase}" == "Succeeded" ]]; then
        break
    fi
    size=0
    if [[ "${phase}" == "Running" ]]; then
        size=$(kubectl --namespace "${K8S_NAMESPACE}" exec "${STAGE_POD_NAME}" -- \
            du -sb "${SHARED_HF_HOME}" 2>/dev/null | awk '{print $1}' || echo 0)
        size=${size:-0}
    fi
    elapsed=$(( $(date +%s) - poll_start ))
    printf '    t=%3ds phase=%-9s cache_bytes=%s\n' "${elapsed}" "${phase}" "${size}"
    samples+=("${elapsed}:${size}")
    sleep 20
done

if (( ${#samples[@]} >= 2 )); then
    first_t=${samples[0]%:*}
    first_s=${samples[0]#*:}
    last_t=${samples[-1]%:*}
    last_s=${samples[-1]#*:}
    dt=$(( last_t - first_t ))
    ds=$(( last_s - first_s ))
    if (( dt > 0 && ds > 0 )); then
        rate_bps=$(( ds / dt ))
        target_bytes=1759218604441
        remaining=$(( target_bytes > last_s ? target_bytes - last_s : 0 ))
        eta=$(( remaining / rate_bps ))
        printf '    estimated cache growth=%s B/s, remaining ETA=%dh%02dm\n' \
            "${rate_bps}" $(( eta / 3600 )) $(( (eta % 3600) / 60 ))
    fi
fi

echo "=== Waiting for staging completion ==="
stage_deadline=$(( $(date +%s) + STAGE_TIMEOUT_SECONDS ))
while true; do
    phase=$(kubectl --namespace "${K8S_NAMESPACE}" get pod "${STAGE_POD_NAME}" \
        -o jsonpath='{.status.phase}' 2>/dev/null || echo Missing)
    case "${phase}" in
        Succeeded)
            break
            ;;
        Failed|Unknown|Missing)
            kubectl --namespace "${K8S_NAMESPACE}" logs "${STAGE_POD_NAME}" || true
            exit 1
            ;;
    esac
    if (( $(date +%s) >= stage_deadline )); then
        echo "ERROR: staging exceeded ${STAGE_TIMEOUT_SECONDS}s" >&2
        exit 1
    fi
    echo "    staging phase=${phase}"
    sleep 60
done
kubectl --namespace "${K8S_NAMESPACE}" logs "${STAGE_POD_NAME}" | tail -n 50
kubectl --namespace "${K8S_NAMESPACE}" delete pod "${STAGE_POD_NAME}" --wait=false

echo "=== Applying four-GB200 pod ==="
kubectl apply -f "${POD_MANIFEST}"
kubectl --namespace "${K8S_NAMESPACE}" wait --for=condition=Ready \
    "pod/${POD_NAME}" --timeout=15m

kubectl --namespace "${K8S_NAMESPACE}" exec "${POD_NAME}" -- \
    test -f "${POD_OUTPUT_ROOT}/.stage_done"
kubectl --namespace "${K8S_NAMESPACE}" exec "${POD_NAME}" -- \
    test -L "${POD_OUTPUT_ROOT}/repo/data/Wan-Syn_77x448x832_600k"

echo "=== Running four-GPU preflight ==="
kubectl --namespace "${K8S_NAMESPACE}" exec "${POD_NAME}" -- \
    bash "${POD_OUTPUT_ROOT}/repo/.agents/k8s/run_preflight.sh" "${POD_OUTPUT_ROOT}"

echo "=== Launching detached 500-step training ==="
TRAIN_CMD="$(cat <<EOF
set -euo pipefail
cd '${POD_OUTPUT_ROOT}'
mkdir -p logs output
nohup bash repo/.agents/k8s/run_train.sh '${POD_OUTPUT_ROOT}' \\
    > logs/train.launcher.out 2>&1 < /dev/null &
TRAIN_PID=\$!
echo \"\${TRAIN_PID}\" > output/train.pid
disown
sleep 2
kill -0 \"\${TRAIN_PID}\"
echo \"[launcher] training process \${TRAIN_PID} is alive\"
EOF
)"
kubectl --namespace "${K8S_NAMESPACE}" exec "${POD_NAME}" -- bash -lc "${TRAIN_CMD}"
TRAIN_PID=$(kubectl --namespace "${K8S_NAMESPACE}" exec "${POD_NAME}" -- \
    cat "${POD_OUTPUT_ROOT}/output/train.pid")

cat <<EOF
=================================================================
Job launched. You can disconnect.
Run name:        ${RUN_NAME}
Pod name:        ${POD_NAME}
Training PID:    ${TRAIN_PID}
Pod output root: ${POD_OUTPUT_ROOT}
Local output:    ${LOCAL_OUTPUT}

Monitor:
  kubectl --kubeconfig ${KUBECONFIG} -n ${K8S_NAMESPACE} exec ${POD_NAME} -- \\
      tail -F ${POD_OUTPUT_ROOT}/logs/train.log

Resume, validate, infer, and download:
  KUBECONFIG=${KUBECONFIG} RUN_NAME=${RUN_NAME} \\
      bash .agents/k8s/resume_k8s.sh
=================================================================
EOF
