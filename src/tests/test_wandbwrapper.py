from src.wandb_wrapper import WandbWrapper, MyRun
from numpy import random

config = {'epochs':10, 'lr':1.e-3}
wandb = WandbWrapper()
wandb.set_params(is_wandb_on=True, is_sweep=False, config=config)
wandb.login()
#run = MyRun()

wandb.init(
    project="GE_test",
    config=config
)

train_dict = {'train/acc':0, 'train_loss': 0.} 

loss = 10.

for epoch in range(100):
    loss -= 0.7
    train_dict['train/acc'] = random.rand()
    train_dict['loss'] = loss
    wandb.log(train_dict)

wandb.finish()
