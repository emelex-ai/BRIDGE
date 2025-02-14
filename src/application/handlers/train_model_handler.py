from src.domain.datamodels import ModelConfig, DatasetConfig, TrainingConfig, WandbConfig
from src.utils.helper_functions import get_run_name, set_seed
from src.application.training import TrainingPipeline
from src.infra.clients.wandb import WandbWrapper
from src.domain.dataset import BridgeDataset
from src.domain.model import Model
import logging
import os


logger = logging.getLogger(__name__)


class TrainModelHandler:
    def __init__(
        self,
        wandb_config: WandbConfig,
        model_config: ModelConfig,
        dataset_config: DatasetConfig,
        training_config: TrainingConfig,
    ):
        """
        Initialize the training handler with configurations.
        """
        self.wandb_config = wandb_config
        self.model_config = model_config
        self.dataset_config = dataset_config
        self.training_config = training_config
        self.wandb_wrapper = None
        self.pipeline = None

    def _initialize_wandb(self, additional_config: dict = None):
        """
        Initialize the WandB wrapper.
        """
        if not self.wandb_wrapper:
            self.wandb_wrapper = WandbWrapper(
                project_name=self.wandb_config.project,
                entity=self.wandb_config.entity,
                is_enabled=self.wandb_config.is_enabled,
                config={
                    **self.model_config.model_dump(),
                    **self.dataset_config.model_dump(),
                    **(additional_config or {}),
                },
            )

    def _setup_pipeline(self):
        """
        Set up the training pipeline and dataset.
        """
        bridge_dataset = BridgeDataset(dataset_config=self.dataset_config, device=self.training_config.device)
        self.pipeline = TrainingPipeline(
            model=Model(
                model_config=self.model_config,
                dataset_config=self.dataset_config,
                device=self.training_config.device,
            ),
            training_config=self.training_config,
            dataset=bridge_dataset,
        )

    def initiate_model_training(self):
        """
        Start the training process.
        """
        self._initialize_wandb()
        self._setup_pipeline()

        run_name = get_run_name(self.training_config.model_artifacts_dir)
        self.wandb_wrapper.start_run(run_name=run_name)

        self.training_config.model_artifacts_dir = os.path.join(self.training_config.model_artifacts_dir, run_name)
        os.makedirs(self.training_config.model_artifacts_dir, exist_ok=True)

        logger.info("Starting training...")
        for metrics in self.pipeline.run_train_val_loop(run_name):
            self.wandb_wrapper.log_metrics(metrics)

        logger.info("Training completed.")

    def initiate_model_pretraining(self):
        """
        Run pretraining across multiple datasets.
        """
        if self.model_config.seed:
            set_seed(self.model_config.seed)

        logger.info("Starting pretraining...")
        for i in range(1, 50):
            self.training_config.model_artifacts_dir = f"model_artifacts/pretraining/{i}"
            self.dataset_config.dataset_filepath = f"data/pretraining/input_data_{i}.pkl"
            os.makedirs(self.training_config.model_artifacts_dir, exist_ok=True)

            self._initialize_wandb()
            self._setup_pipeline()

            run_name = f"pretraining_{i}_" + get_run_name(self.training_config.model_artifacts_dir)
            self.wandb_wrapper.start_run(run_name=run_name)

            logger.info(f"Pretraining on dataset {i}...")
            for metrics in self.pipeline.run_train_val_loop(run_name):
                self.wandb_wrapper.log_metrics(metrics)

            logger.info(f"Pretraining {i} completed.")

    def initiate_sweep(self, sweep_config: dict):
        """
        Create and execute a hyperparameter sweep.
        """
        self._initialize_wandb()
        self.wandb_wrapper.create_sweep(sweep_config)
        self.wandb_wrapper.run_sweep(agent_function=self._run_sweep_iteration)

    def _run_sweep_iteration(self):
        """
        Execute a single iteration of the sweep process.
        """
        self.wandb_wrapper.start_run()

        # Update configs dynamically from wandb sweep
        config_updates = self.wandb_wrapper.get_config()
        for key, value in config_updates.items():
            if hasattr(self.model_config, key):
                setattr(self.model_config, key, value)
            if hasattr(self.training_config, key):
                setattr(self.training_config, key, value)

        self.training_config.model_artifacts_dir = "models/pretraining/1"
        self.dataset_config.dataset_filepath = "data/pretraining/input_data_1.pkl"
        os.makedirs(self.training_config.model_artifacts_dir, exist_ok=True)

        self._setup_pipeline()

        logger.info("Running sweep iteration...")
        for metrics in self.pipeline.run_train_val_loop("sweep_run"):
            self.wandb_wrapper.log_metrics(metrics)

        self.wandb_wrapper.end_run()
