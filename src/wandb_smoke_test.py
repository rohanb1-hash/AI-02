import wandb

with wandb.init(
    project="ai-02-architecture-generalization",
    name="setup-test",
) as run:
    run.log({"test_metric": 1})

print("W&B connection successful!")