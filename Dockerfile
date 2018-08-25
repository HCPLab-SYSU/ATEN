FROM nvidia/cuda:8.0-cudnn6-devel

# Supress warnings about missing front-end. As recommended at:
# http://stackoverflow.com/questions/22466255/is-it-possibe-to-answer-dialog-questions-when-installing-under-docker
ARG DEBIAN_FRONTEND=noninteractive

# Install some dependencies
RUN apt-get update && \
    apt-get install -y \
      apt-utils \
      git \
      curl \
      unzip \
      openssh-client \
      wget \
      build-essential \
      cmake \
      libboost-all-dev \
      libffi-dev \
      libfreetype6-dev \
      libhdf5-dev \
      libjpeg8-dev \
      liblcms2-dev \
      libopenblas-dev \
      liblapack-dev \
      libpng12-dev \
      libssl-dev \
      libtiff5-dev \
      libwebp-dev \
      libzmq3-dev \
      nano \
      pkg-config \
      libavcodec-dev \
      libavformat-dev \
      libswscale-dev \
      libtheora-dev \
      libvorbis-dev \
      libxvidcore-dev \
      libx264-dev \
      yasm \
      libopencore-amrnb-dev \
      libopencore-amrwb-dev \
      libv4l-dev \
      libxine2-dev \
      libtbb-dev \
      libeigen3-dev \
      python3.5 \
      python3.5-dev \
      python3-pip \
      python3-tk \
      zlib1g-dev \
      libprotobuf-dev \
      libleveldb-dev \
      libsnappy-dev \
      libhdf5-serial-dev \
      protobuf-compiler \
      liblmdb-dev \
      libgoogle-glog-dev \
      libatlas-base-dev \
      gfortran \
      libgflags-dev \
      liblapacke-dev \
      libopenblas-dev \
      && \
  apt-get clean && \
  apt-get autoremove && \
  rm -rf /var/lib/apt/lists/*

# upgrade pip(3)
RUN pip3 install --upgrade pip

# python dependencies
RUN pip3 install --no-cache-dir --upgrade Cython numpy pypng scikit-image ipython

# opencv (3.2 specifically)
# ensure dnn is NOT enabled, this will cause problems!
RUN cd ~ && \
    wget -O opencv.zip https://github.com/opencv/opencv/archive/3.2.0.zip && \
    wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/3.2.0.zip && \
    unzip opencv.zip && \
    unzip opencv_contrib.zip && \
    rm -f opencv.zip && \
    rm -f opencv_contrib.zip && \
    cd ~/opencv-3.2.0/ && \
    mkdir build && \
    cd build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D INSTALL_PYTHON_EXAMPLES=OFF \
      -D INSTALL_C_EXAMPLES=OFF \
      -D BUILD_opencv_dnn=OFF \
      -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib-3.2.0/modules \
      -D PYTHON3_EXECUTABLE=/usr/bin/python3 \
      -D BUILD_opencv_python2=OFF \
      -D BUILD_opencv_python3=ON \
      -D BUILD_EXAMPLES=OFF .. && \
    make -j"$(nproc)" && \
    make install -j"$(nproc)" && \
    ldconfig && \
    cd ~ && \
    rm -rf opencv-3.2.0 && \
    rm -rf opencv_contrib-3.2.0

# Tensorflow 1.4.1 - GPU
RUN pip3 install --no-cache-dir --upgrade \
    "https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.4.1-cp35-cp35m-linux_x86_64.whl"

# Install pymongo
RUN pip3 install --no-cache-dir --upgrade keras

# aten
RUN cd /root && \
    git clone --depth=1 https://github.com/HCPLab-SYSU/ATEN.git

# aten - convGRU
RUN cd /root/ATEN/keras_convGRU && \
    python3 setup.py install

# aten - flow_warp
RUN cd `python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())'` && \
  cd tensorflow/stream_executor/cuda && \
  curl -O https://raw.githubusercontent.com/tensorflow/tensorflow/master/third_party/toolchains/gpus/cuda/cuda/cuda_config.h && \
  cd /root/ATEN/ops && \
  sed -i 's/python/python3/' Makefile && \
  sed -i 's/sm_52/sm_61/' Makefile && \
  sed -i 's/ltensorflow_framework/ltensorflow_framework \-D_GLIBCXX_USE_CXX11_ABI\=0/' Makefile && \
  make

WORKDIR "/root"
CMD ["/bin/bash"]
