#!/usr/bin/env bash
# Bring up the Kubernetes sandbox: minikube, monitoring, and the three workloads.
# Idempotent -- safe to re-run.
set -euo pipefail

PROFILE="${MINIKUBE_PROFILE:-cost-opt}"
# Raise on slow networks: the monitoring images are pulled inside the cluster.
HELM_TIMEOUT="${HELM_TIMEOUT:-10m}"
NS="cost-opt-sandbox"
MON_NS="monitoring"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if ! minikube status -p "$PROFILE" >/dev/null 2>&1; then
  echo "==> starting minikube ($PROFILE)"
  minikube start -p "$PROFILE" --driver=docker --memory=4096 --cpus=4
fi
kubectl config use-context "$PROFILE" >/dev/null

echo "==> monitoring (kube-prometheus-stack)"
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts >/dev/null 2>&1 || true
helm upgrade --install kps prometheus-community/kube-prometheus-stack \
  --namespace "$MON_NS" --create-namespace \
  -f "$ROOT/k8s/monitoring-values.yaml" --wait --timeout "$HELM_TIMEOUT"

echo "==> sandbox workloads"
# minikube has its own image store. If the host already has the workload image,
# copy it in rather than pulling it again from inside the cluster.
if docker image inspect python:3.12-alpine >/dev/null 2>&1; then
  minikube -p "$PROFILE" image load python:3.12-alpine
fi
kubectl apply -f "$ROOT/k8s/namespace.yaml"
# Generated from workloads/*.py so the Docker and Kubernetes sandboxes run
# identical scripts.
kubectl create configmap workload-scripts -n "$NS" \
  --from-file="$ROOT/workloads/" --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -f "$ROOT/k8s/workloads.yaml"
kubectl rollout status -n "$NS" deploy --timeout=5m

cat <<MSG

Ready. Expose Prometheus to the agent (leave this running in its own terminal):

  kubectl port-forward -n $MON_NS svc/kps-kube-prometheus-stack-prometheus 9091:9090

MSG
