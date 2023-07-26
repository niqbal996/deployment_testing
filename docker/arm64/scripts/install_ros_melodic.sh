#!/bin/bash
apt update
apt install -y software-properties-common gnupg2 curl
sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
apt update && apt install -y ros-melodic-ros-base
apt install -y python-rosdep python3-wstool build-essential
apt install -y python3-rospkg
apt install -y ros-melodic-rostopic ros-melodic-rospy ros-melodic-roslaunch
apt install -y python-rosdep && rosdep init && rosdep update
echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
apt-get autoremove
