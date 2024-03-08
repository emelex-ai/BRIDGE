import subprocess

subprocess.run(["python", "-m", "src.main", "--sweep", "sweep5.yaml", "--max_nb_steps", "5", "--test", "--which_dataset", "100"])
