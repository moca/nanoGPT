# Config for training GPT-2 (50M) in ~ 1h for 1B tokens
# $ torchrun --standalone --nproc_per_node=4 train.py config/train_gpt2_baby.py
wandb_log = False
wandb_project = 'gpt2-baby'
wandb_run_name='gpt2-50M-g5-12xlarge'

# these make the total batch size be ~0.245M
# 12 batch size * 1024 block size * 5 gradaccum * 4 GPUs = 245760
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 5 * 4

# Baby GPT2 with 50M Params
n_layer = 8
n_head = 8
n_embd = 512
dropout = 0.08

# this makes total number of tokens to be 1B
max_iters = 4200
lr_decay_iters = 4200

# eval stuff
eval_interval = 500
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1
out_dir = 'out-baby-gpt2-50M'