# ConnTextUL
./utilities
Emelex's repository for the ConnTextUL project.

2023-06-01
On the mac, I reset ulimit: "ulimit -n 1024". Wandb 0.15.3 is running the test program. 
To list the number of open files: `lsof`

2023-06-13
I do not understand. When inputting a sequence to the decoder, should it not be
  phon[0] + padding ([33])?    BoS ([31])
