import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.application.interface import BridgeInference

CHECKPOINT_PATH = "model_artifacts/2025_02_09_run_001/model_epoch_5.pth"

if __name__ == "__main__":
    app = BridgeInference(checkpoint_path=CHECKPOINT_PATH)
    app.launch(share=True)
