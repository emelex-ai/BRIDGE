from src.application.training import TrainingPipeline
from src.domain.dataset import ConnTextULDataset
from src.domain.datamodels import ModelConfig
from src.domain.model.model import Model
import torch as pt
import torch


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

        phon_pred = torch.argmax(logits["phon"], dim=1)
        phon_true = phonology["targets"]

        # Create a mask for valid positions (not padding)
        phon_valid_mask = phon_true != 2
        masked_phon_true = phon_true[phon_valid_mask]
        masked_phon_pred = phon_pred[phon_valid_mask]

        # Feature-wise accuracy
        correct_features = (masked_phon_pred == masked_phon_true).sum()
        phon_feature_accuracy = correct_features.float() / phon_valid_mask.sum().float()

        # Phoneme-wise accuracy
        phoneme_wise_mask = phon_pred == phon_true
        phoneme_wise_accuracy = phoneme_wise_mask.all(dim=-1).sum() / (masked_phon_true.shape[0] / phon_true.shape[-1])

        # Phoneme word accuracy
        word_accuracies = [word[target != 2].all().int() for word, target in zip(phoneme_wise_mask, phon_true)]
        phon_word_accuracy = sum(word_accuracies) / len(word_accuracies)

        return {
            "loss": loss.item(),
            "letter_wise_accuracy": letter_wise_accuracy.item(),
            "word_wise_accuracy": orth_word_accuracy.item(),
            "phon_feature_accuracy": phon_feature_accuracy.item(),
            "phoneme_wise_accuracy": phoneme_wise_accuracy.item(),
            "phon_word_accuracy": phon_word_accuracy.item(),
        }
