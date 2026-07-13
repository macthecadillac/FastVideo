#!/usr/bin/env bash
# Submit CPU staging plus a cluster-resident transition to four-GB200 training.
# This script performs cluster mutations when invoked; do not run it until the
# user separately approves the launch.

set -euo pipefail

WORKSTATION_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${WORKSTATION_ROOT}"

KUBECONFIG="${KUBECONFIG:-/home/sandbox/.kube/config}"
export KUBECONFIG
K8S_NAMESPACE="${K8S_NAMESPACE:-vllm}"
BRANCH="${BRANCH:-issue/775-tdm}"
COMMIT="${COMMIT:?COMMIT must be set to the approved full 40-character commit SHA}"
if [[ ! "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "ERROR: COMMIT must be a full lowercase 40-character commit SHA" >&2
    exit 1
fi
RUN_NAME="${RUN_NAME:-tdm-bsz4-500-k8s-$(date +%s)}"
POD_NAME="${POD_NAME:-${RUN_NAME}}"
STAGE_POD_NAME="${STAGE_POD_NAME:-${POD_NAME}-stage}"
SUPERVISOR_JOB_NAME="${SUPERVISOR_JOB_NAME:-${POD_NAME}-supervisor}"
SUPERVISOR_CONFIGMAP="${SUPERVISOR_CONFIGMAP:-${POD_NAME}-launch}"
SUPERVISOR_TIMEOUT_SECONDS="${SUPERVISOR_TIMEOUT_SECONDS:-172800}"
RESTART_STAGING="${RESTART_STAGING:-0}"
PVC_NAME="${PVC_NAME:-lustre-pvc-vllm}"
LOCAL_OUTPUT="${LOCAL_OUTPUT:-${WORKSTATION_ROOT}/outputs/issue-775-tdm/k8s/${RUN_NAME}}"
POD_OUTPUT_ROOT="/workspace/run/issue-775/${RUN_NAME}"
SHARED_HF_HOME="/workspace/run/issue-775/shared-cache/hf"

if [[ "${RESTART_STAGING}" != "0" && "${RESTART_STAGING}" != "1" ]]; then
    echo "ERROR: RESTART_STAGING must be 0 or 1" >&2
    exit 1
fi
for command in kubectl python3; do
    command -v "${command}" >/dev/null 2>&1 || {
        echo "ERROR: required command '${command}' not found" >&2
        exit 1
    }
done
for name in "${POD_NAME}" "${STAGE_POD_NAME}" "${SUPERVISOR_JOB_NAME}" "${SUPERVISOR_CONFIGMAP}"; do
    if (( ${#name} > 63 )); then
        echo "ERROR: Kubernetes resource name '${name}' exceeds 63 characters" >&2
        exit 1
    fi
done
kubectl --namespace "${K8S_NAMESPACE}" get nodes -o name >/dev/null

mkdir -p "${LOCAL_OUTPUT}/manifests"
POD_MANIFEST="${LOCAL_OUTPUT}/manifests/${POD_NAME}.yaml"
POD_JSON="${LOCAL_OUTPUT}/manifests/${POD_NAME}.json"
STAGE_MANIFEST="${LOCAL_OUTPUT}/manifests/${STAGE_POD_NAME}.yaml"
SUPERVISOR_MANIFEST="${LOCAL_OUTPUT}/manifests/${SUPERVISOR_JOB_NAME}.yaml"
CONFIGMAP_MANIFEST="${LOCAL_OUTPUT}/manifests/${SUPERVISOR_CONFIGMAP}.yaml"

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
    SUPERVISOR_CONFIGMAP="${SUPERVISOR_CONFIGMAP}" \
    SUPERVISOR_JOB_NAME="${SUPERVISOR_JOB_NAME}" \
    SUPERVISOR_TIMEOUT_SECONDS="${SUPERVISOR_TIMEOUT_SECONDS}" \
        python3 .agents/k8s/_envsubst.py < "${source}" > "${destination}"
}

render_manifest .agents/k8s/stage-pod.yaml "${STAGE_MANIFEST}"
render_manifest .agents/k8s/pod.yaml "${POD_MANIFEST}"
render_manifest .agents/k8s/supervisor.yaml "${SUPERVISOR_MANIFEST}"
kubectl create --dry-run=client -f "${STAGE_MANIFEST}" >/dev/null
kubectl create --dry-run=client -f "${POD_MANIFEST}" -o json > "${POD_JSON}"
kubectl create --dry-run=client -f "${SUPERVISOR_MANIFEST}" >/dev/null
kubectl --namespace "${K8S_NAMESPACE}" create configmap "${SUPERVISOR_CONFIGMAP}" \
    --from-file=pod.json="${POD_JSON}" --dry-run=client -o yaml > "${CONFIGMAP_MANIFEST}"

cat <<EOF
=== K8s TDM launch (issue 775) ===
Namespace:        ${K8S_NAMESPACE}
Cluster:          $(kubectl config current-context)
Run name:         ${RUN_NAME}
Stage pod:        ${STAGE_POD_NAME}
Supervisor Job:   ${SUPERVISOR_JOB_NAME}
GPU pod:          ${POD_NAME}
Commit:           ${COMMIT}
PVC:              ${PVC_NAME}
Shared HF cache:  ${SHARED_HF_HOME}
Pod output root:  ${POD_OUTPUT_ROOT}
Local output:     ${LOCAL_OUTPUT}
==================================
EOF

if kubectl --namespace "${K8S_NAMESPACE}" get pod "${POD_NAME}" -o name >/dev/null 2>&1; then
    echo "ERROR: GPU pod ${POD_NAME} already exists" >&2
    exit 1
fi
if kubectl --namespace "${K8S_NAMESPACE}" get pod "${STAGE_POD_NAME}" -o name >/dev/null 2>&1; then
    if [[ "${RESTART_STAGING}" != "1" ]]; then
        echo "ERROR: staging pod ${STAGE_POD_NAME} exists; set RESTART_STAGING=1 to replace it" >&2
        exit 1
    fi
    echo "=== Deleting existing staging pod ==="
    kubectl --namespace "${K8S_NAMESPACE}" delete pod "${STAGE_POD_NAME}" \
        --wait=true --timeout=5m
fi

kubectl --namespace "${K8S_NAMESPACE}" delete job "${SUPERVISOR_JOB_NAME}" \
    --ignore-not-found --wait=true --timeout=5m
kubectl apply --dry-run=server -f "${STAGE_MANIFEST}" >/dev/null
kubectl apply --dry-run=server -f "${POD_MANIFEST}" >/dev/null
kubectl apply --dry-run=server -f "${CONFIGMAP_MANIFEST}" >/dev/null
kubectl apply --dry-run=server -f "${SUPERVISOR_MANIFEST}" >/dev/null
echo "Server-side dry-run accepted staging, supervisor, and GPU resources."

kubectl apply -f "${CONFIGMAP_MANIFEST}"
kubectl apply -f "${POD_MANIFEST}"
kubectl apply -f "${SUPERVISOR_MANIFEST}"
kubectl apply -f "${STAGE_MANIFEST}"

cat <<EOF
=================================================================
Cluster-resident launch submitted. The workstation can disconnect.
Run name:       ${RUN_NAME}
Stage pod:      ${STAGE_POD_NAME}
Supervisor Job: ${SUPERVISOR_JOB_NAME}
GPU pod:        ${POD_NAME}

The GPU pod is held outside the scheduler by a staging gate. The supervisor
waits for an atomic stage marker for commit ${COMMIT}, then removes that gate.
The GPU pod runs preflight and training itself after scheduling.

Monitor:
  kubectl --kubeconfig ${KUBECONFIG} -n ${K8S_NAMESPACE} get pods,jobs \
      -l issue=775
  kubectl --kubeconfig ${KUBECONFIG} -n ${K8S_NAMESPACE} logs -f \
      job/${SUPERVISOR_JOB_NAME}
=================================================================
EOF
