#!/bin/bash

python -m ipdb ./train.py \
    -data ./deen.onmt.pt.train.pt \
    -save_model ./onmt_models/deen.onmt.simple \
    -layer 1 \
    -rnn_size 1024 \
    -word_vec_size 512 \
    -batch_size 256 \
    -epochs 500 \
    -optim adam \
    -learning_rate 0.001 \
    -learning_rate_decay 1. \
    -max_grad_norm 1. \
    -gpus 1 \
    -log_interval 50 \
    -brnn \

