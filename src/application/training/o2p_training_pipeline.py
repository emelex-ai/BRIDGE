from src.application.training import TrainingPipeline
from src.domain.dataset import ConnTextULDataset
from src.domain.datamodels import ModelConfig
from src.domain.model.model import Model
import torch as pt


class O2PModelPipeline(TrainingPipeline):

    def __init__(self, model: Model, model_config: ModelConfig, dataset: ConnTextULDataset):
        super().__init__(model, model_config, dataset)

    def forward(self, batch):
        orthography = batch["orthography"].to(self.device)
        phonology = batch["phonology"].to(self.device)
        print(phonology)
        return self.model(
            orthography["enc_input_ids"],
            orthography["enc_pad_mask"],
            phonology["dec_input_ids"],
            phonology["dec_pad_mask"],
        )

    def compute_loss(self, logits, batch):
        phonology = batch["phonology"].to(self.device)
        phon_loss = pt.nn.CrossEntropyLoss(ignore_index=2)(logits["phon"], phonology["targets"])
        return phon_loss
