#!/bin/bash
apt update 
apt install -y wget 
python3 -m pip install -v "protobuf<3.20"
wget https://nvidia.box.com/shared/static/pmsqsiaw4pg9qrbeckcbymho6c01jj4z.whl -O onnxruntime_gpu-1.11.0-cp36-cp36m-linux_aarch64.whl
python3 -m pip install onnxruntime_gpu-1.11.0-cp36-cp36m-linux_aarch64.whl
# install opencv 
apt install -y libgtk2.0-0
wget https://repo.download.nvidia.com/jetson/common/pool/main/libo/libopencv/libopencv_4.1.1-2-gd5a58aa75_arm64.deb
dpkg -i libopencv_4.1.1-2-gd5a58aa75_arm64.deb
wget -i https://repo.download.nvidia.com/jetson/common/pool/main/libo/libopencv-python/libopencv-python_4.1.1-2-gd5a58aa75_arm64.deb
dpkg -i libopencv-python_4.1.1-2-gd5a58aa75_arm64.deb