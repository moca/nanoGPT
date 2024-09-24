# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:

wandb_log = False
wandb_project = 'gpt2-baby'
wandb_run_name='gpt2-50M-g5-12xlarge-2nodes'

# these make the total batch size be ~0.5M in 2 g5.12xnodes
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491520
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 5 * 8 

# Baby GPT2 with 50M Params
n_layer = 8
n_head = 8
n_embd = 512
dropout = 0.08

# this makes total number of tokens to be 1B
max_iters = 500 # 2200
lr_decay_iters = 500 #2200
warmup_iters = 100 

# eval stuff
eval_interval = 50 # 500
eval_iters = 20 # 200
log_interval = 10

# weight decay
weight_decay = 1e-1

out_dir = 'out-baby-gpt2-50M'