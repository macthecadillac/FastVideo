#!/usr/bin/env bash
# Launch the issue-775 TDM training job on the
# oci-bth-aiaccelerator-prd005 Kubernetes cluster.
#
# This is the launch-time driver. It:
#   1. Renders and applies the pod manifest.
#   2. Waits for the pod to start, then for the init container to start
#      its dataset download.
#   3. Polls the HF cache size every 20s for up to 90s to estimate the
#      Wan-Syn dataset download rate. Reports the estimated total init
#      time and projected wall-clock for the 500 training steps.
#   4. Once the init container finishes, exec's the training script
#      inside the pod under `nohup ... &` so it survives the
#      kubectl exec returning. The training process's PID is written
#      to ${POD_OUTPUT_ROOT}/output/train.pid.
#   5. Confirms the training process is alive, prints a clear
#      "you can disconnect" banner with the pod name, run name, and
#      reconnect instructions, and exits.
#
# This script does NOT wait for training to complete. Use
# resume_k8s.sh in a later session to monitor, run inference, and
# download artifacts.
#
# Usage:
#   KUBECONFIG=/home/toolbox/.kube/config \
#   K8S_NAMESPACE=vllm \
#   RUN_NAME=tdm_bsz4_500_k8s_$(date +%s) \
#       bash .agents/k8s/run_k8s.sh
#
# Environment overrides (all optional):
#   KUBECONFIG        default /home/toolbox/.kube/config
#   K8S_NAMESPACE     default vllm
#   BRANCH            default issue/775-tdm
#   COMMIT            default HEAD of BRANCH
#   RUN_NAME          default tdm_bsz4_500_k8s_$(date +%s)
#   POD_NAME          default ${RUN_NAME}
#   PVC_NAME          default lustre-pvc-vllm
#   POLL_SECONDS      default 90   (init-time poll budget)
#   LOCAL_OUTPUT      default ./outputs/issue-775-tdm/k8s/${RUN_NAME}

set -euo pipefail

WORKSTATION_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${WORKSTATION_ROOT}"

KUBECONFIG="${KUBECONFIG:-/home/toolbox/.kube/config}"
export KUBECONFIG
K8S_NAMESPACE="${K8S_NAMESPACE:-vllm}"
BRANCH="${BRANCH:-issue/775-tdm}"
COMMIT="${COMMIT:-$(git rev-parse "origin/${BRANCH}")}"
RUN_NAME="${RUN_NAME:-tdm_bsz4_500_k8s_$(date +%s)}"
POD_NAME="${POD_NAME:-${RUN_NAME}}"
PVC_NAME="${PVC_NAME:-lustre-pvc-vllm}"
POLL_SECONDS="${POLL_SECONDS:-90}"
LOCAL_OUTPUT="${LOCAL_OUTPUT:-${WORKSTATION_ROOT}/outputs/issue-775-tdm/k8s/${RUN_NAME}}"
POD_OUTPUT_ROOT="/workspace/run/issue-775/${RUN_NAME}"

require_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "ERROR: required command '$1' not found on PATH" >&2
        exit 1
    fi
}
require_cmd kubectl
require_cmd git
require_cmd python3
require_cmd tar

kubectl --namespace "${K8S_NAMESPACE}" get node -o name >/dev/null

echo "=== K8s TDM launch (issue 775) ==="
echo "Namespace:        ${K8S_NAMESPACE}"
echo "Cluster:          $(kubectl config current-context)"
echo "Run name:         ${RUN_NAME}"
echo "Pod name:         ${POD_NAME}"
echo "Pod output root:  ${POD_OUTPUT_ROOT}"
echo "Local output:     ${LOCAL_OUTPUT}"
echo "Branch:           ${BRANCH}"
echo "Commit:           ${COMMIT}"
echo "Init poll budget: ${POLL_SECONDS}s"
echo "PVC:              ${PVC_NAME}"
echo "=============================="

mkdir -p "${LOCAL_OUTPUT}/manifests"
POD_MANIFEST="${LOCAL_OUTPUT}/manifests/${POD_NAME}.yaml"
POD_OUTPUT_ROOT="${POD_OUTPUT_ROOT}" \
POD_NAME="${POD_NAME}" \
K8S_NAMESPACE="${K8S_NAMESPACE}" \
BRANCH="${BRANCH}" \
COMMIT="${COMMIT}" \
RUN_NAME="${RUN_NAME}" \
python3 .agents/k8s/_envsubst.py < .agents/k8s/pod.yaml > "${POD_MANIFEST}"
echo "Wrote manifest: ${POD_MANIFEST}"

if kubectl --namespace "${K8S_NAMESPACE}" get pod "${POD_NAME}" -o name >/dev/null 2>&1; then
    echo "ERROR: pod ${POD_NAME} already exists in namespace ${K8S_NAMESPACE}." >&2
    echo "Delete it (kubectl -n ${K8S_NAMESPACE} delete pod ${POD_NAME})" >&2
    echo "or pick a new RUN_NAME." >&2
    exit 1
fi

echo
echo "=== Applying pod ==="
kubectl apply -f "${POD_MANIFEST}"

echo
echo "=== Waiting for pod ${POD_NAME} to be Running ==="
kubectl --namespace "${K8S_NAMESPACE}" wait \
    --for=jsonpath='{.status.phase}'=Running \
    "pod/${POD_NAME}" --timeout=10m

echo
echo "=== Brief init-time poll (${POLL_SECONDS}s) ==="
echo "    Polling ${POD_OUTPUT_ROOT}/cache/hf size every 20s to estimate"
echo "    dataset download rate."
echo

poll_start=$(date +%s)
samples=()
sample_idx=0
poll_deadline=$(( poll_start + POLL_SECONDS ))
while (( $(date +%s) < poll_deadline )); do
    size=$(kubectl --namespace "${K8S_NAMESPACE}" exec "${POD_NAME}" -c stage -- \
        du -sb "${POD_OUTPUT_ROOT}/cache/hf" 2>/dev/null | awk '{print $1}' || echo 0)
    size=${size:-0}
    human=$(numfmt --to=iec --suffix=B "${size}" 2>/dev/null || echo "${size}B")
    elapsed=$(( $(date +%s) - poll_start ))
    printf "    t=%3ds  cache_size=%10s  %s\n" "${elapsed}" "${size}" "${human}"
    samples+=("${elapsed}:${size}")
    sample_idx=$(( sample_idx + 1 ))
    if (( sample_idx >= 5 )); then break; fi
    sleep 20
done
echo
if (( ${#samples[@]} >= 2 )); then
    first_t=${samples[0]%:*}; first_s=${samples[0]#*:}
    last_t=${samples[-1]%:*};  last_s=${samples[-1]#*:}
    dt=$(( last_t - first_t ))
    ds=$(( last_s - first_s ))
    if (( dt > 0 )) && (( ds > 0 )); then
        rate_bps=$(( ds / dt ))
        rate_human=$(numfmt --to=iec --suffix=B/s "${rate_bps}" 2>/dev/null || echo "${rate_bps}B/s")
        target_bytes=$(( 2 * 1024 * 1024 * 1024 * 1024 ))
        eta=$(( (target_bytes - last_s) / rate_bps ))
        eta_human=$(printf '%dh%02dm' $((eta/3600)) $(((eta%3600)/60)))
        echo "    Estimated download rate: ${rate_human}"
        echo "    Estimated dataset ETA:   ~${eta_human} (assuming ~1.6 TiB total)"
    else
        echo "    Cache size did not grow during the poll window; the init"
        echo "    container may still be in the git-clone or pip-install"
        echo "    phase, or the dataset download has not started yet."
    fi
fi

echo
echo "=== Continuing to wait for init container to finish ==="
echo "    You can interrupt with Ctrl-C; the pod will keep running."
echo "    Use resume_k8s.sh later to continue from this state."
echo
kubectl --namespace "${K8S_NAMESPACE}" wait \
    --for=jsonpath='{.status.initContainerStatuses[0].state.terminated.exitCode}'=0 \
    "pod/${POD_NAME}" --timeout=24h

echo
echo "=== Init container finished; pod main container is up ==="
kubectl --namespace "${K8S_NAMESPACE}" get pod "${POD_NAME}" -o wide

echo
echo "=== Launching training inside pod (detached, nohup) ==="
TRAIN_CMD="$(cat <<EOF
set -euo pipefail
cd '${POD_OUTPUT_ROOT}'
mkdir -p logs output
nohup bash repo/.agents/k8s/run_train.sh '${POD_OUTPUT_ROOT}' \\
    > logs/train.launcher.out 2>&1 < /dev/null &
TRAIN_PID=\$!
echo "\${TRAIN_PID}" > output/train.pid
disown
sleep 2
if kill -0 "\${TRAIN_PID}" 2>/dev/null; then
    echo "[launcher] training process \${TRAIN_PID} is alive"
else
    echo "[launcher] training process \${TRAIN_PID} is NOT alive; see logs/train.launcher.out"
    cat logs/train.launcher.out
    exit 1
fi
EOF
)"
kubectl --namespace "${K8S_NAMESPACE}" exec "${POD_NAME}" -- \
    bash -lc "${TRAIN_CMD}"

echo
echo "=== Verifying training is alive ==="
TRAIN_PID=$(kubectl --namespace "${K8S_NAMESPACE}" exec "${POD_NAME}" -- \
    cat "${POD_OUTPUT_ROOT}/output/train.pid" 2>/dev/null || echo "")
if [[ -n "${TRAIN_PID}" ]]; then
    if kubectl --namespace "${K8S_NAMESPACE}" exec "${POD_NAME}" -- \
        kill -0 "${TRAIN_PID}" 2>/dev/null; then
        echo "    training pid ${TRAIN_PID} is alive"
    else
        echo "    WARNING: training pid ${TRAIN_PID} is NOT alive"
        echo "    see ${POD_OUTPUT_ROOT}/logs/train.launcher.out for the launcher log"
    fi
fi

cat <<EOF

=================================================================
  Job launched. You can disconnect.
=================================================================
  Run name:        ${RUN_NAME}
  Pod name:        ${POD_NAME}
  Pod status:      $(kubectl --namespace "${K8S_NAMESPACE}" get pod "${POD_NAME}" -o jsonpath='{.status.phase}' 2>/dev/null)
  Training PID:    ${TRAIN_PID}
  Pod output root: ${POD_OUTPUT_ROOT}
  Local output:    ${LOCAL_OUTPUT}

  To monitor from any new session:
    kubectl --kubeconfig /home/toolbox/.kube/config \\
        -n ${K8S_NAMESPACE} logs -f ${POD_NAME} -c main
    kubectl --kubeconfig /home/toolbox/.kube/config \\
        -n ${K8S_NAMESPACE} exec ${POD_NAME} -- \\
        tail -F ${POD_OUTPUT_ROOT}/logs/train.log

  When you are ready for inference, run:
    KUBECONFIG=/home/toolbox/.kube/config \\
    RUN_NAME=${RUN_NAME} \\
        bash .agents/k8s/resume_k8s.sh
=================================================================
EOF
