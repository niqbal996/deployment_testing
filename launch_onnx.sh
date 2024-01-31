#!/bin/bash
confidence=$1
model=$2
ldconfig
source /opt/ros/noetic/setup.bash
python3 onnx_infer.py --conf $confidence --model $model
