1. command to train DDPM with normalized Toyzero
real: `CUDA_VISIBLE_DEVICES=0 python main_toyzero.py train --batch_size=16 --ckpt_interval=50 --run_name toyzero_normalized-100_real --domain real --max_steps_per_epoch=2000`
fake: `CUDA_VISIBLE_DEVICES=0 python main_toyzero.py train --batch_size=16 --ckpt_interval=50 --run_name toyzero_normalized-100_fake --domain fake --max_steps_per_epoch=2000`

1. command to infer with trained DDPM
``


To-Does:
1. Make normalization parameters (clamp, etc) configurable.

