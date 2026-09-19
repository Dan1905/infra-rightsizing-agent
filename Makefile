# Common tasks. `make help` lists them.
#
# The agent reads BACKEND from .env; pass it here to override for one command:
#   make analyze BACKEND=kubernetes

PYTHON    ?= python3.13
VENV      ?= .venv
RIGHTSIZE := $(VENV)/bin/rightsize
BACKEND_FLAG := $(if $(BACKEND),--backend $(BACKEND),)

COMPOSE := docker compose -f sandbox/docker/docker-compose.yml

.PHONY: help install test up down k8s-up k8s-down port-forward index metrics analyze run audit

help:  ## List targets
	@grep -E '^[a-z0-9-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-13s %s\n", $$1, $$2}'

install:  ## Create the virtualenv and install the package with dev tools
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install -e ".[dev]"

test:  ## Run the test suite (no Docker, cluster, network or model needed)
	$(VENV)/bin/python -m pytest

up:  ## Start the Docker sandbox (cAdvisor, Prometheus, three workloads)
	$(COMPOSE) up -d

down:  ## Stop the Docker sandbox
	$(COMPOSE) down

k8s-up:  ## Start minikube, kube-prometheus-stack and the workloads
	./sandbox/kubernetes/up.sh

k8s-down:  ## Delete the minikube cluster
	minikube delete -p cost-opt

port-forward:  ## Expose the cluster's Prometheus on localhost:9091 (keep running)
	kubectl port-forward -n monitoring svc/kps-kube-prometheus-stack-prometheus 9091:9090

index:  ## Build the policy vector index
	$(RIGHTSIZE) index --rebuild

metrics:  ## Show current metrics for the selected backend
	$(RIGHTSIZE) $(BACKEND_FLAG) metrics

analyze:  ## Propose a plan; never executes
	$(RIGHTSIZE) $(BACKEND_FLAG) analyze

run:  ## Propose, ask for typed approval, then execute
	$(RIGHTSIZE) $(BACKEND_FLAG) run

audit:  ## Show recent runs and decisions
	$(RIGHTSIZE) audit
