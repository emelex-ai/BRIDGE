from typing import Tuple
from src.domain.model import Model
from src.domain.datamodels import DatasetConfig, ModelConfig
from src.domain.dataset.bridge_dataset import BridgeDataset
from src.utils.helper_funtions import load_model_config


class BridgeModelService:
    """
    Handles domain-specific operations:
      - Loading configurations, model weights, and datasets.
      - Performing inference: encoding inputs, generating outputs, and decoding results.
    """

    def __init__(self, checkpoint_path: str) -> None:
        self.checkpoint_path = checkpoint_path
        self.model, self.dataset = self._load_resources()

    def _load_resources(self) -> Tuple[Model, BridgeDataset, BridgeDataset]:
        config_dict = load_model_config(self.checkpoint_path)
        dataset_config = DatasetConfig(**config_dict["dataset_config"])
        model_config = ModelConfig(**config_dict["model_config"])

        # Initialize dataset and model
        dataset = BridgeDataset(dataset_config)
        model = Model(model_config, dataset_config)

        # Load weights and set model to evaluation mode
        model.load_state_dict(config_dict["model_state_dict"])
        model.eval()

        return model, dataset

    def generate_output(self, words: str, pathway: str) -> str:
        if not words.strip():
            return "Please enter words separated by spaces or commas."
        if not pathway:
            return "Please select a pathway."

        # Normalize input: replace spaces with commas and split by comma
        word_list = [w.strip() for w in words.replace(" ", ",").split(",") if w.strip()]
        if not word_list:
            return "Invalid input. Please enter valid words."

        # Encode the input words using the dataset's encoding method.
        encodings = self.dataset.encode(word_list)
        # Generate model output; assume model_output is a dict with token sequences.
        model_output = self.model.generate(encodings, pathway=pathway).model_dump()

        decoded_results = []
        if pathway in ["p2o", "o2o"]:
            orth_tokens = model_output.get("orth_tokens")
            orth_token_list = orth_tokens.tolist() if hasattr(orth_tokens, "tolist") else orth_tokens

            for i, token_seq in enumerate(orth_token_list):
                decoded_word = self.dataset.character_tokenizer.decode([token_seq])
                if i < len(word_list):
                    decoded_results.append(f"{word_list[i]}: {decoded_word}")
                else:
                    decoded_results.append(decoded_word)
        elif pathway in ["o2p", "p2p"]:
            phon_tokens = model_output.get("phon_tokens")
            phon_token_list = phon_tokens.tolist() if hasattr(phon_tokens, "tolist") else phon_tokens

            for i, token_seq in enumerate(phon_token_list):
                decoded_word = self.dataset.phonemizer.decode(token_seq)
                if i < len(word_list):
                    decoded_results.append(f"{word_list[i]}: {decoded_word}")
                else:
                    decoded_results.append(decoded_word)
        else:
            # For pathways returning both orthographic and phonological outputs.
            orth_tokens = model_output.get("orth_tokens")
            phon_tokens = model_output.get("phon_tokens")
            orth_token_list = orth_tokens.tolist() if hasattr(orth_tokens, "tolist") else orth_tokens
            phon_token_list = phon_tokens.tolist() if hasattr(phon_tokens, "tolist") else phon_tokens

            for i, (ot, pt) in enumerate(zip(orth_token_list, phon_token_list)):
                decoded_orth = self.dataset.character_tokenizer.decode([ot])
                decoded_phon = self.dataset.phonemizer.decode(pt)
                if i < len(word_list):
                    decoded_results.append(f"{word_list[i]}: orth: {decoded_orth}, phon: {decoded_phon}")
                else:
                    decoded_results.append(f"orth: {decoded_orth}, phon: {decoded_phon}")

        return "\n".join(decoded_results)
