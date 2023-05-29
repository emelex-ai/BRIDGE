import argparse  # NEW LIBRARY (pip install argparse)
from wandb_wrapper import WandbWrapper
from train import run_code

wandb = WandbWrapper()


def main():
    """ """
    #  Three parameters specific to W&B
    entity = "emelex"
    project = "GE_ConnTextUL"
    is_wandb_enabled = False

    #  Parameters specific to the main code

    config = {
        # "starting_epoch": epoch_num,   # Add it back later once code is debugged
        "CONTINUE": False,
        "num_epochs": 1,
        "batch_size": 128,
        "d_model": 32,
        "nhead": 4,
        "learning_rate": 1.0e-3,
        "train_test_split": 0.8,
        # "id": model_id,  # Add back later once code is debugged
        "common_num_layers": 1,
        # Set to -1 if all the steps should be executed
        "max_nb_steps": 10,   # to speed up testing
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true", default='False')
    args = parser.parse_args()

    print("args.sweep: ", args.sweep)

    if args.sweep == True:
        print("args.sweep")
        wandb.set_params(
            config=config, is_sweep=True, is_wandb_on=is_wandb_enabled
        )  # GE: new function

        # make wandb wrapper accessible globally
        globals().update({"wandb": wandb})
        wandb.login()
        # Is it possible to update a sweep configuration? I'd like the default sweep
        # configuration to contain the parameters of config.
        # GE: suggestion: load different sweeps from files to keep track. 
        sweep_config = {
            "method": "grid",
            "name": "sweep_time_per_epoch",
            "metric": {
                'goal': 'minimize', 
                'name': 'time_per_epoch',
            },
            "parameters": {
                "batch_size": {"values": [32, 64, 128]},
                "d_model": {"values": [16, 32, 64, 128]},
                "common_num_layers": {"values": [1, 2, 4]},
            },
        }

        # Update sweep_config with new_params without overwriting existing parameters:
        for param, value in config.items():
            if param not in sweep_config["parameters"]:
                sweep_config["parameters"][param] = {"values": [value]}

        sweep_id = wandb.sweep(sweep_config, project=project, entity=entity)
        wandb.agent(sweep_id, run_code)
    else:
        print("else")
        wandb.set_params(config=config, is_sweep=False, is_wandb_on=is_wandb_enabled)

        globals().update({"wandb": wandb})
        wandb.login()
        # 🐝 initialise a wandb run
        # I created a new project
        run = wandb.init(
            entity=entity,  # Necessary because I am in multiple teams
            project=project,
            config=config,
        )
        # When 'disabled', the returned run.config is an empty dictionary {}
        if not is_wandb_enabled:
            print("wandb is disabled")
            run = wandb.init(config=config)
            # make sure I config is accessible with the dot notation
            run_code()
        else:
            run_code()


if __name__ == "__main__":
    main()
