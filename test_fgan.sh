#!/bin/bash

python run.py --num_epoch 50000 --num_iter -1 --p 100 --s 50 --sparse_estimation 0 --eps 0.1 \
              --train_size 500 --coord_median_as_origin 0 --contamination gauss_0.0 \
              --loss JSLoss --real_batch_size 1000 --fake_batch_size 1000 \
	          --simultaneous 0 --num_step_d 15 --num_step_g 1 \
		      --d_optimizer sgd --g_optimizer sgd \
              --d_sgd_lr 0.002 --g_sgd_lr 0.002 --g_sgd_momentum 0.9 \
	          --sgd_weight_decay 0.01 \
              --seed 0 \
