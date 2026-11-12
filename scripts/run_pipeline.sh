#!/usr/bin/env bash
# Run every offline stage end to end on the synthetic data.
set -euo pipefail
export MLFLOW_DISABLE_AGENT_HINT=1
python scripts/generate_data.py
python -m budget_assistant.features.build
python -m budget_assistant.model.train
python -m budget_assistant.model.serve --as-of 2025-06-30 revenue treasury it hr
python -m budget_assistant.monitoring.drift
python -m budget_assistant.rag.eval_retrieval
python -m budget_assistant.finetune.benchmark
python -m budget_assistant.experiments.bayesian_ab --simulate 0.62 0.68
python evals/run_evals.py
