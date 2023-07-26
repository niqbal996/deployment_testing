#!/bin/bash
apt update 
apt install -y wget 
python3 -m pip install -v "protobuf>=3.20.2" # for onnxruntime  1.13.0
python3 -m pip install numpy==1.23.0    # See https://github.com/NVIDIA/TensorRT/issues/2567
python3 -m pip install cuda-python
wget https://nvidia.box.com/shared/static/v59xkrnvederwewo2f1jtv6yurl92xso.whl -O onnxruntime_gpu-1.12.1-cp38-cp38-linux_aarch64.whl
python3 -m pip install onnxruntime_gpu-1.12.1-cp38-cp38-linux_aarch64.whl
# OpenCV is installed already in L4t TensorRT containers 
# apt install -y libgtk2.0-0
# wget https://repo.download.nvidia.com/jetson/common/pool/main/libo/libopencv/libopencv_4.1.1-2-gd5a58aa75_arm64.deb
# dpkg -i libopencv_4.1.1-2-gd5a58aa75_arm64.deb
# wget -i https://repo.download.nvidia.com/jetson/common/pool/main/libo/libopencv-python/libopencv-python_4.1.1-2-gd5a58aa75_arm64.deb
# dpkg -i libopencv-python_4.1.1-2-gd5a58aa75_arm64.deb