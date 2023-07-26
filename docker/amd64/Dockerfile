FROM nvcr.io/nvidia/tensorrt:23.04-py3

WORKDIR /workspace/ros_5g/
COPY . /workspace/ros_5g/
USER root
# Install python dependencies
RUN --mount=target=/var/lib/apt/lists,type=cache,sharing=locked \
    --mount=target=/var/cache/apt,type=cache,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean && \
    apt update && \
    apt install -y wget software-properties-common gnupg2 curl ffmpeg libsm6 libxext6 && \ 
    apt-add-repository universe && \
    apt-add-repository multiverse && \ 
    apt-add-repository restricted && \
    python3 -m pip install --upgrade pip && \
    apt autoremove

# Run pip install using the download cache
RUN --mount=type=cache,target=/root/.cache \
    python3 -m pip install onnxruntime-gpu imutils opencv-python opencv-contrib-python torchvision==0.14.1 torch==1.13.1

# Install ROS and use the download cache
RUN --mount=target=/var/lib/apt/lists,type=cache,sharing=locked \
    --mount=target=/var/cache/apt,type=cache,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean && \
    sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' && \
    apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654 && \
    apt-get update && \
    apt-get install -y ros-noetic-ros-base && \
    apt install -y python3-rosdep python3-wstool build-essential  && \
    apt install -y python3-rospkg && \
    apt install -y ros-noetic-rostopic ros-noetic-rospy ros-noetic-roslaunch ros-noetic-sensor-msgs && \
    apt install -y python3-rosdep && rosdep init && rosdep update && \
    apt autoremove && \
    echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc

# ENTRYPOINT ["/workspace/ros_5g/docker/amd64/entrypoint.sh"]