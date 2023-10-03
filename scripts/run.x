python -m src.main \
    --nb_samples 0 \
    --num_epochs 20 \
    --num_layers 4 \
    --batch_size_train 32 \
    --batch_size_val 32 \
    --d_model 128 \
	--d_embedding 1 \
    --nhead 4 \
    --pathway o2p \
    --which_dataset 10000 \
    --test \
    --train_test_split 0.8 \
    --save_every 2 \
    --seed 1337 \
    --learning_rate 0.0001 \
    --project myGE  \
	--wandb

#python -m mainy \
#    --wandb
# Try and duplicate a case run by Nathan that produced good accuracy
#	--test \
