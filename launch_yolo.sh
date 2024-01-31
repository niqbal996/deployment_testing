#!/bin/bash
confidence=$1
# model=$2
ldconfig
source /opt/ros/noetic/setup.bash
python3 trt_infer_yolo.py --conf $confidence
