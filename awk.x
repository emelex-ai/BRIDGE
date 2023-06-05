# Utility to remove wandb processes from ps -elf

ps -elf | grep 'wandb' | awk '{print $4}' | xargs kill -9
