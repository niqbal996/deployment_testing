#!/bin/bash
set -e

ROS_DISTRO="noetic"
ROS_IP=$(hostname -I | awk -F' ' '{print $1}')
ROS_MASTER_URI=http://$ROS_IP:11311

# setup ros environment
source "/opt/ros/$ROS_DISTRO/setup.bash"
echo "source /opt/ros/$ROS_DISTRO/setup.bash" >> $HOME/.bashrc

python3 /workspace/ros_5g/onnx_infer.py
