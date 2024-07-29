import pytest


@pytest.fixture
def required_args():
    """Return the list of required arguments"""

    # modify to include everythign that is required
    return ["device", "project", "num_epochs", "batch_size_train", "batch_size_val", "num_phon_enc_layers", "num_orth_enc_layers", "learning_rate", "d_model", "nhead", "wandb", "train_test_split", "sweep_filename", "d_embedding", "seed", "model_path", "pathway", "save_every", "dataset_filename", "max_nb_steps", "test_filenames"]