#!/bin/bash

python mmd.py --num_epoch 500 --p 100 --eps 0.1 --train_size 50000 --contamination dirac_3. \
              --real_batch_size 1000 --fake_batch_size 1000 \
              --g_sgd_lr 0.0002 --g_sgd_momentum 0.9 --g_sgd_normalize 1 \
              --sigma 10. \
              --seed 0 \
              --debug 0 \
              --test 0
