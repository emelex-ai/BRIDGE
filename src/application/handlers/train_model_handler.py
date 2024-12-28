from src.domain.datamodels import ModelConfig, DatasetConfig, TrainingConfig, WandbConfig
from src.utils.helper_funtions import get_next_run_name
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
        self.wandb_wrapper = WandbWrapper(
            project_name=wandb_config.project,
            entity=wandb_config.entity,
            is_enabled=wandb_config.is_enabled,
            config={**model_config.model_dump(), **dataset_config.model_dump()},
        )
        bridge_dataset = BridgeDataset(dataset_config=dataset_config, device=training_config.device)

        self.pipeline = TrainingPipeline(
            model=Model(
                model_config=model_config,
                dataset_config=dataset_config,
                device=training_config.device,
            ),
            training_config=training_config,
            dataset=bridge_dataset,
        )

    def initiate_model_training(self):
        run_name = get_next_run_name(self.pipeline.training_config.model_artifacts_dir)
        self.wandb_wrapper.start_run(run_name=run_name)
        self.pipeline.training_config.model_artifacts_dir = (
            f"{self.pipeline.training_config.model_artifacts_dir}/{run_name}/"
        )

        for metrices in self.pipeline.run_train_val_loop(run_name):
            if self.wandb_wrapper.is_enabled:
                self.wandb_wrapper.log_metrics(metrices)

    def initiate_model_pretraining(self):
        # TODO: implement initiate_model_pretraining
        pass

    def initiate_model_weep_training(self):
        # TODO: implement initiate_model_weep_training
        pass
