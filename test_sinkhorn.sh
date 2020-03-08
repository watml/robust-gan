#!/bin/bash

python sinkhorn.py --num_epoch 1000 --p 1 --eps 0.1 --train_size 50000 --contamination gaussian_0.5 \
                   --real_batch_size 1000 --fake_batch_size 1000 \
                   --g_sgd_lr 0.0001 --g_sgd_momentum 0.9 --g_sgd_normalize 0 \
                   --lam 0.1 --sinkhorn_max_iter 1000 --const 1e-6 --thres 0.001 \
                   --seed 0 \
                   --debug 0 \
                   --test 0
