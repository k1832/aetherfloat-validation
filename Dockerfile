# Use Ubuntu 22.04 as the base image
FROM ubuntu:22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# 1. Install dependencies for building Yosys and OpenSTA
RUN apt-get update && apt-get install -y \
    build-essential \
    clang \
    bison \
    flex \
    libreadline-dev \
    gawk \
    tcl-dev \
    libffi-dev \
    git \
    graphviz \
    xdot \
    pkg-config \
    python3 \
    libboost-system-dev \
    libboost-python-dev \
    libboost-filesystem-dev \
    zlib1g-dev \
    cmake \
    swig \
    python3-dev \
    curl \
    libeigen3-dev \
    autoconf \
    automake \
    libtool \
    && rm -rf /var/lib/apt/lists/*

# 2. Build and Install Yosys (shallow clone to save disk)
WORKDIR /usr/src
RUN git clone --depth 1 --recurse-submodules --shallow-submodules \
        https://github.com/YosysHQ/yosys.git yosys \
    && cd yosys \
    && make config-gcc \
    && make -j$(nproc) \
    && make install \
    && cd /usr/src && rm -rf yosys

# 3. Build and Install CUDD (required by OpenSTA)
RUN git clone --depth 1 https://github.com/The-OpenROAD-Project/cudd.git \
    && cd cudd \
    && autoreconf -fi \
    && ./configure --prefix=/usr/local \
    && make -j$(nproc) \
    && make install \
    && cd /usr/src && rm -rf cudd

# 4. Build and Install OpenSTA (shallow clone to save disk)
RUN git clone --depth 1 https://github.com/The-OpenROAD-Project/OpenSTA.git \
    && cd OpenSTA \
    && mkdir build && cd build \
    && cmake .. -DCUDD_DIR=/usr/local \
    && make -j$(nproc) \
    && make install \
    && cd /usr/src && rm -rf OpenSTA

# 5. Verify installations
RUN yosys -V && sta -version

# 6. Final Setup
WORKDIR /workspace
CMD ["bash"]
