import wandb
import logging
from src.application.shared import Singleton

logger = logging.getLogger(__name__)


class WandbWrapper(Singleton):
    def __init__(self, project_name, entity=None, config=None, is_enabled=True):
        """
        Initialize the W&B wrapper.

        :param project_name: Name of the W&B project.
        :param entity: W&B entity (username or team).
        :param config: Dictionary of hyperparameters to log.
        :param is_enabled: Flag to enable or disable W&B integration.
        """
        self.project_name = project_name
        self.entity = entity
        self.config = config or {}
        self.is_enabled = is_enabled
        self.run = None
        self.sweep_id = None

        if self.is_enabled:
            logger.info("WandbWrapper initialized with project_name: %s, entity: %s", project_name, entity)
        else:
            logger.info("WandbWrapper initialized in disabled mode.")

    def start_run(self, run_name=None, config_updates=None):
        """
        Start a W&B run.

        :param run_name: Optional name for the run.
        :param config_updates: Dictionary of additional config updates.
        """
        if not self.is_enabled:
            logger.info("W&B is disabled. Skipping start_run.")
            return

        logger.info("Starting W&B run with name: %s", run_name)
        self.run = wandb.init(
            project=self.project_name,
            entity=self.entity,
            name=run_name,
            config={**self.config, **(config_updates or {})},
        )
        logger.info("W&B run started")

    def log_metrics(self, metrics, step=None):
        """
        Log metrics to the current W&B run.

        :param metrics: Dictionary of metrics to log.
        :param step: Optional step number.
        """
        if not self.is_enabled:
            logger.debug("W&B is disabled. Skipping log_metrics.")
            return

        if self.run is None:
            logger.error("Attempted to log metrics without an active W&B run")
            raise RuntimeError("W&B run not initialized. Call start_run() first.")
        logger.debug("Logging metrics: %s at step: %s", metrics, step)
        wandb.log(metrics, step=step)

    def log_artifact(self, artifact_name, artifact_type, files):
        """
        Log an artifact (e.g., model checkpoints, datasets).

        :param artifact_name: Name of the artifact.
        :param artifact_type: Type of the artifact (e.g., 'model', 'dataset').
        :param files: Path(s) to the files to include in the artifact.
        """
        if not self.is_enabled:
            logger.info("W&B is disabled. Skipping log_artifact.")
            return

        if self.run is None:
            logger.error("Attempted to log artifact without an active W&B run")
            raise RuntimeError("W&B run not initialized. Call start_run() first.")

        logger.info("Logging artifact: %s of type: %s", artifact_name, artifact_type)
        artifact = wandb.Artifact(artifact_name, type=artifact_type)
        if isinstance(files, str):
            artifact.add_file(files)
        elif isinstance(files, list):
            for file in files:
                artifact.add_file(file)
        else:
            logger.error("Invalid files parameter: %s", files)
            raise ValueError("Files should be a string or a list of strings.")

        self.run.log_artifact(artifact)
        logger.info("Artifact logged successfully")

    def end_run(self):
        """
        End the current W&B run.
        """
        if not self.is_enabled:
            logger.info("W&B is disabled. Skipping end_run.")
            return

        if self.run is not None:
            logger.info("Ending W&B run")
            wandb.finish()
            self.run = None
            logger.info("W&B run ended successfully")

    def create_sweep(self, sweep_config):
        """
        Create a W&B sweep.

        :param sweep_config: Dictionary containing the sweep configuration.
        """
        if not self.is_enabled:
            logger.info("W&B is disabled. Skipping create_sweep.")
            return

        logger.info("Creating W&B sweep with configuration: %s", sweep_config)
        self.sweep_id = wandb.sweep(sweep_config, project=self.project_name, entity=self.entity)
        logger.info("Sweep created with ID: %s", self.sweep_id)

    def run_sweep(self, agent_function, count=None):
        """
        Run a W&B sweep using a specified agent function.

        :param agent_function: Function to execute for each sweep run.
        :param count: Number of runs to execute (default: None for unlimited).
        """
        if not self.is_enabled:
            logger.info("W&B is disabled. Skipping run_sweep.")
            return

        if not self.sweep_id:
            logger.error("No sweep ID found. Create a sweep first using create_sweep().")
            raise RuntimeError("Sweep not initialized. Call create_sweep() first.")

        logger.info("Running W&B sweep with ID: %s", self.sweep_id)
        wandb.agent(self.sweep_id, function=agent_function, count=count)
        logger.info("Sweep execution completed")

    def get_config(self):
        """
        Retrieve the current WandB config.
        :return: Dictionary containing configuration parameters.
        """
        if not self.is_enabled:
            logger.warning("W&B is disabled. Config retrieval not supported.")
            return {}

        if self.run is None:
            logger.error("Attempted to retrieve config without an active W&B run.")
            raise RuntimeError("W&B run not initialized. Call start_run() first.")

        logger.debug("Retrieving W&B config.")
        return dict(wandb.config)
