#!/bin/bash
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################

# set gpu id to use
export CUDA_VISIBLE_DEVICES=0

# set python path according to your actual environment
pythonpath='python3'

# step 4: train model, you can find the model file in ./models/ after training
${pythonpath} ./main.py
