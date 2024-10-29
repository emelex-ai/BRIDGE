from src.application.training import TrainingPipeline
from src.domain.dataset import ConnTextULDataset
from src.domain.datamodels import TrainingConfig
from src.domain.model.model import Model
import torch as pt
import torch


class O2PModelPipeline(TrainingPipeline):

    def __init__(self, model: Model, training_config: TrainingConfig, dataset: ConnTextULDataset):
        super().__init__(model, training_config, dataset)

    def forward(self, orthography, phonology):

        return self.model(
            orthography["enc_input_ids"],
            orthography["enc_pad_mask"],
            phonology["dec_input_ids"],
            phonology["dec_pad_mask"],
        )

    def compute_loss(self, logits, orthography, phonology):
        phon_loss = pt.nn.CrossEntropyLoss(ignore_index=2)(logits["phon"], phonology["targets"])
        return phon_loss

    def compute_metrics(self, logits, orthography, phonology, loss):
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
            "phon_feature_accuracy": phon_feature_accuracy.item(),
            "phon_phoneme_wise_accuracy": phoneme_wise_accuracy.item(),
            "phon_word_accuracy": phon_word_accuracy,
        }
