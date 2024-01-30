#!/bin/bash
ldconfig
source /opt/ros/noetic/setup.bash
python3 onnx_infer.py
