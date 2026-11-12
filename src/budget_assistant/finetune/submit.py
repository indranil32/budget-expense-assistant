"""Upload training files and start a fine-tuning job on Azure OpenAI / Foundry.

    python -m budget_assistant.finetune.submit --base-model <fine-tunable model> [--wait]

Check which base models your region supports for fine-tuning in the Foundry model catalog.
After the job succeeds, deploy the fine-tuned model in Foundry and set
AZURE_OPENAI_FINETUNED_DEPLOYMENT. Fine-tuned deployments are billed per hour while
deployed — delete the deployment when you have finished benchmarking.
"""

from __future__ import annotations

import argparse
import time

from ..llm import _azure_client
from .prepare import OUT, write


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--suffix", default="claims-classifier")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()

    print("Prepared:", write())
    client = _azure_client()
    files = {}
    for name in ("train", "valid"):
        with (OUT / f"{name}.jsonl").open("rb") as fh:
            files[name] = client.files.create(file=fh, purpose="fine-tune").id
    job = client.fine_tuning.jobs.create(
        model=args.base_model,
        training_file=files["train"],
        validation_file=files["valid"],
        suffix=args.suffix,
        seed=13,
        hyperparameters={"n_epochs": args.epochs},
    )
    print(f"Started fine-tuning job {job.id} on {args.base_model}")
    while args.wait and job.status not in ("succeeded", "failed", "cancelled"):
        time.sleep(60)
        job = client.fine_tuning.jobs.retrieve(job.id)
        print(f"  status={job.status}")
    if job.status == "succeeded":
        print(f"Fine-tuned model: {job.fine_tuned_model}  -> deploy it in Foundry, then run finetune.benchmark --live")


if __name__ == "__main__":
    main()
