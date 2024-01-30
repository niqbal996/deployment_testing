#!/bin/bash
ldconfig
source /opt/ros/noetic/setup.bash
python3 trt_infer_yolo.py
