from src.domain.datamodels import ModelConfig, DatasetConfig
from src.application.training import O2PModelPipeline, OP2OPModelPipeline, P2OModelPipeline
from src.domain.model import O2PModel, OP2OPModel, P2OModel
from src.domain.dataset import ConnTextULDataset


class TrainModelHandler:
    def __init__(self, model_config: ModelConfig, dataset_config: DatasetConfig):
        self.model_config = model_config
        self.dataset_config = dataset_config
        conntextul_dataset = ConnTextULDataset(dataset_config=dataset_config)

    def handle_model_training(self):
        if self.model_config.pathway == "o2p":
            model = O2PModel(model_config=self.model_config, dataset_config=self.dataset_config)
            pipeline = O2PModelPipeline(model=model, model_config=self.model_config, dataset_config=self.dataset_config)
        elif self.model_config.pathway == "p2o":
            model = P2OModel(model_config=self.model_config, dataset_config=self.dataset_config)
            pipeline = P2OModelPipeline()
        elif self.model_config.pathway == "op2op":
            model = OP2OPModel(model=model, model_config=self.model_config, dataset_config=self.dataset_config)
            pipeline = OP2OPModelPipeline()

        pipeline.run_train_val_loop()
