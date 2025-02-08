import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import gradio as gr
from src.domain.model import Model
from src.domain.datamodels import DatasetConfig, ModelConfig
from src.utils.helper_funtions import load_model_config
from src.domain.dataset.bridge_dataset import BridgeDataset


checkpoint_path = "model_artifacts/2025_01_26_run_005/model_epoch_19.pth"
config_dict = load_model_config(checkpoint_path)

dataset_config = DatasetConfig(**config_dict["dataset_config"])
model_config = ModelConfig(**config_dict["model_config"])
dataset = BridgeDataset(dataset_config)
model = Model(model_config, dataset_config)

model.load_state_dict(config_dict["model_state_dict"])
model.eval()

def generate_output(words, pathway):
    if not words.strip():
        return "Please enter words separated by spaces or commas."
    if not pathway:
        return "Please select a pathway."

    words = words.replace(" ", ",")
    word_list = [word.strip() for word in words.split(",") if word.strip()]
    if not word_list:
        return "Invalid input. Please enter valid words."

    encodings = dataset.encode(word_list)
    output = model.generate(encodings, pathway=pathway)
    return output

iface = gr.Interface(
    fn=generate_output,
    inputs=[
        gr.Textbox(placeholder="Enter words separated by spaces or commas", label="Input Words", interactive=True),
        gr.Radio(
            choices=["o2p", "p2o", "op2op", "p2p", "o2o"], label="Select Transformation Pathway", interactive=True
        ),
    ],
    outputs=gr.Textbox(label="Generated Output", interactive=True),
    title="Bridge Model Inference Tool",
    description=(
        "Welcome to the Bridge Model Inference Tool!\n"
        "\n"
        "**How to use the app:**\n"
        "1. Enter words separated by spaces or commas in the input box (e.g., 'special cat hat').\n"
        "2. Choose a pathway from the available options.\n"
        "3. Click the submit button to generate the output.\n"
        "\n"
        "**Pathway descriptions:**\n"
        "- **'o2p'**: Orthographic to Phonological transformation.\n"
        "- **'p2o'**: Phonological to Orthographic transformation.\n"
        "- **'op2op'**: Orthographic-Phonological to Orthographic-Phonological transformation.\n"
        "- **'p2p'**: Phonological to Phonological transformation.\n"
        "- **'o2o'**: Orthographic to Orthographic transformation.\n"
    )
)

iface.launch(share=True)
