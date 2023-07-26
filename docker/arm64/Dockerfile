FROM nvcr.io/nvidia/l4t-tensorrt:r8.5.2.2-devel

WORKDIR /opt/
COPY . /opt/
USER root
RUN /opt/install_dependencies_jetpack5.0.sh
RUN /opt/install_ros_noetic.sh
COPY ./entrypoint.sh /home/docker/ 