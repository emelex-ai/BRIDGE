from src.application.training import TrainingPipeline
from src.domain.dataset import ConnTextULDataset
from src.domain.datamodels import ModelConfig
from src.domain.model.model import Model
import torch as pt


class OP2OPModelPipeline(TrainingPipeline):

    def __init__(self, model: Model, model_config: ModelConfig, dataset: ConnTextULDataset):
        super().__init__(model, model_config, dataset)

    def forward(self, batch):
        orthography = batch["orthography"].to(self.device)
        return self.model(
            orthography["enc_input_ids"],
            orthography["enc_pad_mask"],
            orthography["dec_input_ids"],
            orthography["dec_pad_mask"],
        )

    def compute_loss(self, logits, batch):
        orthography = batch["orthography"].to(self.device)
        orth_loss = pt.nn.CrossEntropyLoss(ignore_index=4)(logits["orth"], orthography["enc_input_ids"][:, 1:])
        return orth_loss
