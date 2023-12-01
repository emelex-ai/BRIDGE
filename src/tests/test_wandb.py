import wandb
from numpy import random

config = {"epochs": 10, "lr": 1.0e-3}

wandb.init(project="GE_test", config=config)

train_dict = {}  #'train/acc':0, 'train_loss': 0.}
val_dict = {}  #'valid/acc':0, 'valid_loss': 0.}
loss = 10.0

for epoch in range(100):
    for steps in range(20):
        loss *= 0.999
        train_dict["train/acc"] = random.rand()
        train_dict["train/loss"] = loss
        wandb.log(train_dict)

    val_dict["valid/acc"] = 20.0 + random.rand()
    val_dict["valid/loss"] = 5.0 - random.rand()
    wandb.log(val_dict)

wandb.finish()
