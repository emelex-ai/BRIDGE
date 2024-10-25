from src.application.training import TrainingPipeline
import torch as pt


class O2PModelPipeline(TrainingPipeline):
    def forward(self, batch):
        orthography = batch["orthography"].to(self.device)
        phonology = batch["phonology"].to(self.device)
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
