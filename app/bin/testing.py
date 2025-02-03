
import pickle

import test
from src.domain.model import Model
from src.domain.datamodels import DatasetConfig, ModelConfig
import torch
from src.application.handlers import (
    ModelConfigHandler,
    DatasetConfigHandler,
    TrainingConfigHandler,
    WandbConfigHandler,
    LoggingConfigHandler,
    TrainModelHandler,
)
import os
import sys

import torch
from src.domain.datamodels.dataset_config import DatasetConfig
from src.domain.datamodels.encodings import BridgeEncoding
from src.domain.datamodels.model_config import ModelConfig
from src.domain.datamodels.training_config import TrainingConfig
from src.domain.dataset.bridge_dataset import BridgeDataset
from src.domain.model.model import Model
from src.application.training.phon_metrics import calculate_phoneme_wise_accuracy, calculate_cosine_distance, calculate_euclidean_distance, calculate_phon_word_accuracy
import sys
model_config: ModelConfig = ModelConfigHandler(config_filepath='app/config/model_config.yaml').get_config()
training_config: TrainingConfig = TrainingConfigHandler(config_filepath='app/config/training_config.yaml').get_config()
# Load the test dataset



for i in range(2, 14):
    for j in range(0, 1000, 50):
        try:
            final_data = {}
            pretraining_dataset_config: DatasetConfig = DatasetConfigHandler(config_filepath='app/config/dataset_config.yaml').get_config()
            pretraining_dataset_config.dataset_filepath = f"data/pretraining/input_data_{i}.pkl"
            pretraining_dataset_config.max_orth_seq_len = None
            pretraining_dataset_config.max_phon_seq_len = None
            pretraining_dataset_config.orthographic_vocabulary_size = None
            pretraining_dataset_config.phonological_vocabulary_size = None
            pretraining_dataset = BridgeDataset(dataset_config=pretraining_dataset_config, device=training_config.device)
            print(pretraining_dataset_config.dimension_phon_repr)
            print(pretraining_dataset_config.max_phon_seq_len, pretraining_dataset_config.max_orth_seq_len)
            model = Model(model_config=model_config, dataset_config=pretraining_dataset_config, device=training_config.device)
            training_config.checkpoint_path = f"models/pretraining/{i}/model_epoch_{j}.pth"
            checkpoint = torch.load(training_config.checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Pretraining dataset
            output = model.generate(pretraining_dataset.data, pathway='p2p')
            phon_pred = output.phon_vecs
            phon_true = pretraining_dataset.data.phonological.targets
            phon_pred = torch.stack([torch.stack(i) for i in phon_pred])
            phon_probs = torch.stack([torch.stack(i) for i in output.phon_probs])
            # Mask the padding tokens
            phon_valid_mask = phon_true != 2
            masked_phon_true = phon_true[phon_valid_mask]
            masked_phon_pred = phon_pred[phon_valid_mask]
            phoneme_wise_mask = phon_pred == phon_true
            phoneme_wise_accuracy = calculate_phoneme_wise_accuracy(
                phon_true, masked_phon_true, phoneme_wise_mask
            )
            word_wise_accuracy = calculate_phon_word_accuracy(
                phon_true, phoneme_wise_mask
            )
            cosine_accuracy = calculate_cosine_distance(phon_true, phon_pred)
            euclidean_distance = calculate_euclidean_distance(phon_true, phon_pred)
            final_data['pretraining'] = {
                # 'phon_probabilities': phon_probs,
                'phon_predictions': phon_pred,
                'phon_targets': phon_true,
                'words': pretraining_dataset.words,
                'phoneme_wise_accuracy': phoneme_wise_accuracy,
                'word_wise_accuracy': word_wise_accuracy,
                'cosine_accuracy': cosine_accuracy,
                'euclidean_distance': euclidean_distance
            }
            print(phoneme_wise_accuracy, word_wise_accuracy, cosine_accuracy, euclidean_distance)
            
            
            pretraining_dataset_config = pretraining_dataset_config
            pretraining_dataset_config.max_orth_seq_len = None
            pretraining_dataset_config.max_phon_seq_len = None
            pretraining_dataset_config.orthographic_vocabulary_size = None
            pretraining_dataset_config.phonological_vocabulary_size = None
            pretraining_dataset_config.dataset_filepath = 'data/tests/fry_1980.pkl'
            test_dataset = BridgeDataset(dataset_config=pretraining_dataset_config, device=training_config.device)
            print(test_dataset.dataset_config.json())
            
            output = model.generate(test_dataset.data, pathway='p2p')
            phon_pred = output.phon_vecs
            phon_true = test_dataset.data.phonological.targets
            phon_pred = torch.stack([torch.stack(i) for i in phon_pred])
            phon_probs = torch.stack([torch.stack(i) for i in output.phon_probs])
            # Mask the padding tokens
            phon_valid_mask = phon_true != 2
            masked_phon_true = phon_true[phon_valid_mask]
            masked_phon_pred = phon_pred[phon_valid_mask]
            phoneme_wise_mask = phon_pred == phon_true
            phoneme_wise_accuracy = calculate_phoneme_wise_accuracy(
                phon_true, masked_phon_true, phoneme_wise_mask
            )
            word_wise_accuracy = calculate_phon_word_accuracy(
                phon_true, phoneme_wise_mask
            )
            cosine_accuracy = calculate_cosine_distance(phon_true, phon_pred)
            euclidean_distance = calculate_euclidean_distance(phon_true, phon_pred)
            final_data['fry_1980'] = {
                # 'phon_probabilities': phon_probs,
                'phon_predictions': phon_pred,
                'phon_targets': phon_true,
                'words': test_dataset.words,
                'phoneme_wise_accuracy': phoneme_wise_accuracy,
                'word_wise_accuracy': word_wise_accuracy,
                'cosine_accuracy': cosine_accuracy,
                'euclidean_distance': euclidean_distance
            }
            print(phoneme_wise_accuracy, word_wise_accuracy, cosine_accuracy, euclidean_distance)
            pretraining_dataset_config.orthographic_vocabulary_size = None
            pretraining_dataset_config.phonological_vocabulary_size = None
            pretraining_dataset_config.dataset_filepath = 'data/tests/ewfg.pkl'
            test_dataset_2 = BridgeDataset(dataset_config=pretraining_dataset_config, device=training_config.device)
            
            
            output = model.generate(test_dataset_2.data, pathway='p2p')
            phon_pred = output.phon_vecs
            phon_true = test_dataset_2.data.phonological.targets
            phon_pred = torch.stack([torch.stack(i) for i in phon_pred])
            phon_probs = torch.stack([torch.stack(i) for i in output.phon_probs])
            # Mask the padding tokens
            phon_valid_mask = phon_true != 2
            masked_phon_true = phon_true[phon_valid_mask]
            masked_phon_pred = phon_pred[phon_valid_mask]
            phoneme_wise_mask = phon_pred == phon_true
            phoneme_wise_accuracy = calculate_phoneme_wise_accuracy(
                phon_true, masked_phon_true, phoneme_wise_mask
            )
            word_wise_accuracy = calculate_phon_word_accuracy(
                phon_true, phoneme_wise_mask
            )
            cosine_accuracy = calculate_cosine_distance(phon_true, phon_pred)
            euclidean_distance = calculate_euclidean_distance(phon_true, phon_pred)
            final_data['ewfg'] = {
                # 'phon_probabilities': phon_probs,
                'phon_predictions': phon_pred,
                'phon_targets': phon_true,
                'words': test_dataset_2.words,
                'phoneme_wise_accuracy': phoneme_wise_accuracy,
                'word_wise_accuracy': word_wise_accuracy,
                'cosine_accuracy': cosine_accuracy,
                'euclidean_distance': euclidean_distance
            }
            print(phoneme_wise_accuracy, word_wise_accuracy, cosine_accuracy, euclidean_distance)
            with open(f"results/pretraining_{i}_epoch_{j}.pkl", 'wb') as f:
                pickle.dump(final_data, f)
            
        except Exception as e:
            print(e)
            continue