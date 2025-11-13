#!/bin/bash
set -e

python train_nvmd_cnn_bilstm_together.py --mode-col Mode_2  --epoch 20
python train_nvmd_cnn_bilstm_together.py --mode-col Mode_3  --epoch 20
python train_nvmd_cnn_bilstm_together.py --mode-col Mode_4  --epoch 20
python train_nvmd_cnn_bilstm_together.py --mode-col Mode_5  --epoch 20
python train_nvmd_cnn_bilstm_together.py --mode-col Mode_6  --epoch 20
python train_nvmd_cnn_bilstm_together.py --mode-col Mode_7  --epoch 20
python train_nvmd_cnn_bilstm_together.py --mode-col Mode_8  --epoch 20
python train_nvmd_cnn_bilstm_together.py --mode-col Mode_9  --epoch 20
python train_nvmd_cnn_bilstm_together.py --mode-col Mode_10 --epoch 20
python train_nvmd_cnn_bilstm_together.py --mode-col Mode_11 --epoch 20
python train_nvmd_cnn_bilstm_together.py --mode-col Mode_12 --epoch 20
