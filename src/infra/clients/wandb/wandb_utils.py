import wandb
import logging
from src.infra.clients.wandb.wandb_wrapper import WandbWrapper

logger = logging.getLogger(__name__)


def read_wandb_history(entity, project, run):
    """Utility to read wandb history."""
    logger.info("Entering read_wandb_history function.")
    api = wandb.Api()
    run_id = run.id
    run_name = run.name

    logger.info(f"Run ID: {run_id}, Run Name: {run_name}")

    assert WandbWrapper().is_wandb_on, "Wandb must be active"

    # Retrieve the run from wandb
    run = api.run(f"{entity}/{project}/{run_id}")
    logger.info(f"Run config: {run.config}")
