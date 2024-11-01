#!/usr/bin/bash

export NPU_VISIBLE_DEVICES=0
export ASCEND_RT_VISIBLE_DEVICES=0

python -u 01_train_rectified_flow_npu.py
