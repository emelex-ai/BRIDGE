from src.application.training import TrainingPipeline
import torch as pt


class P2OModelPipeline(TrainingPipeline):
    def forward(self, batch):
        phonology = batch["phonology"].to(self.device)
        orthography = batch["orthography"].to(self.device)
        return self.model(
            phonology["enc_input_ids"],
            phonology["enc_pad_mask"],
            orthography["dec_input_ids"],
            orthography["dec_pad_mask"],
        )

    def compute_loss(self, logits, batch):
        orthography = batch["orthography"].to(self.device)
        orth_loss = pt.nn.CrossEntropyLoss(ignore_index=4)(logits["orth"], orthography["enc_input_ids"][:, 1:])
        return orth_loss
