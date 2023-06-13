# ConnTextUL
./utilities
Emelex's repository for the ConnTextUL project.

2023-06-01
On the mac, I reset ulimit: "ulimit -n 1024". Wandb 0.15.3 is running the test program. 
To list the number of open files: `lsof`

Branch: fix_sweeps
pip install protobuf==3.20.0 (see GPT4)
python -m src.main --sweep sweep4.yaml --num_epochs 1 --which_dataset 100
 (why is nb epochs wrong?)
