#!/bin/bash
set -e

ROS_DISTRO="noetic"
ROS_IP=$(hostname -I | awk -F' ' '{print $1}')
ROS_MASTER_URI=http://$ROS_IP:11311

# setup ros environment
source "/opt/ros/$ROS_DISTRO/setup.bash"
echo "source /opt/ros/$ROS_DISTRO/setup.bash" >> $HOME/.bashrc

roscore &
rosbag play --loop ~/data/2022-06-21-10-42-12.bag aver_01/camera_color/image_raw:=camera/color/image_raw

exec "$@"
