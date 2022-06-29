FROM nvcr.io/nvidia/tensorrt:21.11-py3

# Timezone Configuration
ENV TZ=Europe/Moscow
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    apt-utils \
    net-tools \
    mesa-utils \
    gnupg2 \
    wget \
    curl \
    git \
    mc \
    nano \
    cmake \
    gcc \
    cmake-curses-gui \
    build-essential \
    python3.8 \
    && rm -rf /var/lib/apt/lists/*


ENV DEBIAN_FRONTEND=noninteractive

# ROS install
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu focal main" > /etc/apt/sources.list.d/ros-latest.list'
RUN curl -sSL 'http://keyserver.ubuntu.com/pks/lookup?op=get&search=0xC1CF6E31E6BADE8868B172B4F42ED6FBAB17C654' | apt-key add -
RUN apt-get update && apt-get install -y \
    ros-noetic-robot \
    ros-noetic-rosconsole \
    ros-noetic-realsense2-camera\
    ros-noetic-pcl-ros \
    ros-noetic-image-pipeline
RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc

RUN apt-get update && apt-get install -y \
    python3-opencv ca-certificates python3-dev git wget sudo ninja-build

# RUN wget https://bootstrap.pypa.io/get-pip.py && \
#     python3 get-pip.py && \
#     rm get-pip.py

RUN apt-get update && \
    apt-get install -y python3-pip \
    libpcl-dev \
    python3-catkin-tools \
    python3-dev \
    libopencv-dev


RUN pip3 install \
    cmake \
    gdown \
    pandas \
    rospkg \
    scipy \
    pytimedinput \
    faiss-cpu \
    -U albumentations --no-binary qudida,albumentations \
    netifaces \
    shapely \
    torchfile \
    opencv-python \
    pyfastnoisesimd \
    rapidfuzz \
    Pillow \
    numpy && \
    export ROS_HOSTNAME=localhost 



RUN pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu115
# RUN pip3 install Pillow numpy torch torchvision -f https://download.pytorch.org/whl/cu113/torch_stable.html --upgrade

RUN pip3 install 'git+https://github.com/facebookresearch/fvcore' 
#opencv-python==4.5.2.54

# RUN apt-get update && apt-get install -y python3-catkin-tools python3-dev libopencv-dev
EXPOSE 11311


# install CV bridge for python3
RUN mkdir -p /cv_bridge_ws/src && \
    cd /cv_bridge_ws/src && \
    git clone https://github.com/IvDmNe/vision_opencv.git && \
    cd /cv_bridge_ws && \
    catkin config \
    -DPYTHON_EXECUTABLE=/usr/bin/python3 \
    -DPYTHON_INCLUDE_DIR=/usr/include/python3.8 \
    -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.8.so \
    -DCMAKE_BUILD_TYPE=Release \
    -DSETUPTOOLS_DEB_LAYOUT=OFF \
    -Drosconsole_DIR=/opt/ros/noetic/share/rosconsole/cmake \
    -Drostime_DIR=/opt/ros/noetic/share/rostime/cmake \
    -Droscpp_traits_DIR=/opt/ros/noetic/share/roscpp_traits/cmake \
    -Dstd_msgs_DIR=/opt/ros/noetic/share/std_msgs/cmake \
    -Droscpp_serialization_DIR=/opt/ros/noetic/share/roscpp_serialization/cmake \
    -Dmessage_runtime_DIR=/opt/ros/noetic/share/message_runtime/cmake \
    -Dgeometry_msgs_DIR=/opt/ros/noetic/share/geometry_msgs/cmake \
    -Dsensor_msgs_DIR=/opt/ros/noetic/share/sensor_msgs/cmake \
    -Dcpp_common_DIR=/opt/ros/noetic/share/cpp_common/cmake && \
    cd src && git clone https://github.com/ros/catkin.git &&  cd .. && \
    catkin config --install && \
    catkin build cv_bridge && \
    echo "source /cv_bridge_ws/devel/setup.bash --extend" >> ~/.bashrc


# Install MMCV and MMDetection
RUN pip3 install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu115/torch1.11/index.html && \
    git clone https://github.com/open-mmlab/mmdetection.git /mmdetection && \
    pip3 install -r /mmdetection/requirements/build.txt && \
    pip3 install --no-cache-dir -e /mmdetection


RUN echo "source /ws/devel/setup.bash --extend" >> ~/.bashrc
ENV TORCH_HOME='/ws'
WORKDIR /ws

# install tensorRT

# wget https://developer.nvidia.com/compute/machine-learning/tensorrt/secure/8.2.4/tars/tensorrt-8.2.4.2.linux.x86_64-gnu.cuda-11.4.cudnn8.2.tar.gz
# RUN python3 -m pip install --upgrade setuptools pip
# RUN pip3 install nvidia-pyindex && \
#     pip3 install nvidia-tensorrt
# RUN pip3 install pycuda




## install mmdeploy
# ENV ONNXRUNTIME_DIR=/root/workspace/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}
ENV TENSORRT_DIR=/workspace/tensorrt
ARG VERSION
RUN git clone https://github.com/open-mmlab/mmdeploy /workspace/mmdeploy &&\
    cd /workspace/mmdeploy &&\
    if [ -z ${VERSION} ] ; then echo "No MMDeploy version passed in, building on master" ; else git checkout tags/v${VERSION} -b tag_v${VERSION} ; fi &&\
    git submodule update --init --recursive &&\
    mkdir -p build &&\
    cd build &&\
    cmake -DMMDEPLOY_TARGET_BACKENDS="trt" .. &&\
    make -j$(nproc) &&\
    cd .. &&\
    pip install -e .

### build sdk
ARG PPLCV_VERSION=0.6.2
RUN git clone https://github.com/openppl-public/ppl.cv.git /workspace/ppl.cv&&\
    cd /workspace/ppl.cv &&\
    git checkout tags/v${PPLCV_VERSION} -b v${PPLCV_VERSION} &&\
    ./build.sh cuda

ENV BACKUP_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda-11.5/compat/lib.real/:$LD_LIBRARY_PATH

RUN apt-get install -y libspdlog-dev

RUN cd /workspace/mmdeploy &&\
    rm -rf build/CM* build/cmake-install.cmake build/Makefile build/csrc &&\
    mkdir -p build && cd build &&\
    cmake .. \
        -DMMDEPLOY_BUILD_SDK=ON \
        -DCMAKE_CXX_COMPILER=g++ \
        -Dpplcv_DIR=/workspace/ppl.cv/cuda-build/install/lib/cmake/ppl \
        -DTENSORRT_DIR=${TENSORRT_DIR} \
        -DMMDEPLOY_BUILD_SDK_PYTHON_API=ON \
        -DMMDEPLOY_TARGET_DEVICES="cuda;cpu" \
        -DMMDEPLOY_TARGET_BACKENDS="trt" \
        -DMMDEPLOY_CODEBASES=all &&\
    make -j$(nproc) && make install &&\
    cd install/example  && mkdir -p build && cd build &&\
    cmake -DMMDeploy_DIR=/workspace/mmdeploy/build/install/lib/cmake/MMDeploy .. &&\
    make -j$(nproc) && export SPDLOG_LEVEL=warn &&\
    if [ -z ${VERSION} ] ; then echo "Built MMDeploy master for GPU devices successfully!" ; else echo "Built MMDeploy version v${VERSION} for GPU devices successfully!" ; fi

ENV LD_LIBRARY_PATH="/workspace/mmdeploy/build/lib:${BACKUP_LD_LIBRARY_PATH}"

# install detectron2
# set FORCE_CUDA because during `docker build` cuda is not accessible
ENV FORCE_CUDA="1"
# This will by default build detectron2 for all common cuda architectures and take a lot more time,
# because inside `docker build`, there is no way to tell which architecture will be used.
ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

RUN cd /workspace &&\
    git clone https://github.com/facebookresearch/detectron2 detectron2_repo &&\
    pip3 install -e detectron2_repo

