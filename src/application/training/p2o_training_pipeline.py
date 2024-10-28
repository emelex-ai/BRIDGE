from src.application.training import TrainingPipeline
from src.domain.dataset import ConnTextULDataset
from src.domain.datamodels import ModelConfig
from src.domain.model.model import Model
import torch as pt
import torch


class P2OModelPipeline(TrainingPipeline):

    def __init__(self, model: Model, model_config: ModelConfig, dataset: ConnTextULDataset):
        super().__init__(model, model_config, dataset)

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

    def compute_metrics(self, logits, orthography, phonology, loss):
        orth_pred = torch.argmax(logits["orth"], dim=1)
        orth_true = orthography["enc_input_ids"][:, 1:]

        # Create a mask for valid positions (not padding)
        orth_valid_mask = orth_true != 4
        masked_orth_true = orth_true[orth_valid_mask]
        masked_orth_pred = orth_pred[orth_valid_mask]

        # Letter-wise accuracy
        correct_matches = (masked_orth_pred == masked_orth_true).sum()
        letter_wise_accuracy = correct_matches.float() / orth_valid_mask.sum().float()

        # Word-wise accuracy
        orth_pred[~orth_valid_mask] = 4
        word_wise_mask = orth_pred == orth_true
        orth_word_accuracy = word_wise_mask.all(dim=1).float().mean()

        return {
            "loss": loss.item(),
            "letter_wise_accuracy": letter_wise_accuracy.item(),
            "word_wise_accuracy": orth_word_accuracy.item(),
        }
