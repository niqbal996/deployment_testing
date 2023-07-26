#!/bin/bash
apt install -y software-properties-common gnupg2 curl
sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
apt-get update
apt-get install ros-noetic-ros-base
apt install python3-rosdep python3-wstool build-essential 
apt install -y python3-rospkg
apt install -y ros-noetic-rostopic ros-noetic-rospy ros-noetic-roslaunch 
apt install -y python3-rosdep && rosdep init && rosdep update
echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc