from src.application.training import O2PModelPipeline, OP2OPModelPipeline, P2OModelPipeline
from src.domain.datamodels import ModelConfig, DatasetConfig, TrainingConfig
from src.domain.model import O2PModel, OP2OPModel, P2OModel
from src.domain.dataset import ConnTextULDataset


class TrainModelHandler:

    def __init__(self, model_config: ModelConfig, dataset_config: DatasetConfig, training_config: TrainingConfig):

        conntextul_dataset = ConnTextULDataset(dataset_config=dataset_config)

        if model_config.pathway == "o2p":
            self.pipeline = O2PModelPipeline(
                model=O2PModel(model_config=model_config, dataset_config=conntextul_dataset.dataset_config),
                training_config=training_config,
                dataset=conntextul_dataset,
            )
        elif model_config.pathway == "p2o":
            self.pipeline = P2OModelPipeline(
                model=P2OModel(model_config=model_config, dataset_config=conntextul_dataset.dataset_config),
                training_config=training_config,
                dataset=conntextul_dataset,
            )
        elif model_config.pathway == "op2op":
            self.pipeline = OP2OPModelPipeline(
                model=OP2OPModel(model_config=model_config, dataset_config=conntextul_dataset.dataset_config),
                training_config=training_config,
                dataset=conntextul_dataset,
            )

    def initiate_model_training(self):
        self.pipeline.run_train_val_loop()
