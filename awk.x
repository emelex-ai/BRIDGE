# Utility to remove wandb processes from ps -elf

sudo ps -elf | grep 'wandb' | awk '{print $2}' | xargs kill -9
sudo ps -elf | grep 'python' | awk '{print $2}' | xargs kill -9
