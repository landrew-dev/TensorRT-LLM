install no local commands:
```
export HOME=/home/scratch.jdaw_coreai/landrew

pip3 install torch==2.7.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip3 install --upgrade pip setuptools
pip3 install tensorrt_llm

# install cuda-toolkit
mkdir -p $HOME/cuda-local && cd $HOME/cuda-local
wget https://developer.download.nvidia.com/compute/cuda/12.9.0/local_installers/cuda_12.9.0_575.51.03_linux.run
chmod +x cuda_12.9.0_*.run
./cuda_12.9.0_*.run --silent --toolkit --installpath=$HOME/cuda-12.9

export CUDA_HOME=$HOME/cuda-12.9
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

link to wheels: https://pypi.nvidia.com/tensorrt-llm/

install local commands:
```
export HOME=/home/scratch.jdaw_coreai/landrew

# newer version of cmake for trt build
CMAKE_VER=3.30.4
ARCH=x86_64
cd $HOME/opt || mkdir -p $HOME/opt && cd $HOME/opt
wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VER}/cmake-${CMAKE_VER}-linux-${ARCH}.tar.gz
tar -xzf cmake-${CMAKE_VER}-linux-${ARCH}.tar.gz
export PATH="$HOME/opt/cmake-${CMAKE_VER}-linux-${ARCH}/bin:$PATH"

# install cuda-toolkit
mkdir -p $HOME/cuda-local && cd $HOME/cuda-local
wget https://developer.download.nvidia.com/compute/cuda/12.9.1/local_installers/cuda_12.9.1_575.57.08_linux.run
mkdir -p $HOME/tmp
export TMPDIR=$HOME/tmp
export TMP=$HOME/tmp
export CUDA_LOG_DIR=$HOME/tmp
export CUDA_INSTALLER_LOG=$HOME/tmp/cuda-installer.log
sh $HOME/cuda-local/cuda_12.9.1_575.57.08_linux.run --tmpdir=$HOME/tmp --extract=$HOME/cuda-extract
cd $HOME/cuda-extract
find . -name cuda-installer -type f
./cuda-linux.12.9.1_575.57.08/cuda-installer --silent --toolkit --installpath=$HOME/cuda-12.9

export CUDA_HOME=$HOME/cuda-12.9
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# install nccl
mkdir -p $HOME/nccl && cd $HOME/nccl
# NOTE: in reality you have to download this locally and then scp, because the raw link requires an authenticated session
# e.g. scp ~/Downloads/nccl_2.28.3-1+cuda13.0_x86_64.txz landrew@computelab-frontend-6:/home/scratch.jdaw_coreai/landrew/nccl/
# wget https://developer.nvidia.com/downloads/compute/machine-learning/nccl/secure/2.28.3/agnostic/x64/nccl_2.28.3-1+cuda13.0_x86_64.txz
tar xvf nccl_2.28.3-1+cuda13.0_x86_64.txz
export NCCL_ROOT=$HOME/nccl/nccl_2.28.3-1+cuda13.0_x86_64
export CMAKE_PREFIX_PATH=$NCCL_ROOT
export CPATH=$NCCL_ROOT/include:$CPATH
export LD_LIBRARY_PATH=$NCCL_ROOT/lib:$LD_LIBRARY_PATH

# install tensorrt
mkdir -p $HOME/tensorrt && cd $HOME/tensorrt
# NOTE: in reality you have to download this locally and then scp, because the raw link requires an authenticated session
# e.g. scp ~/Downloads/TensorRT-10.13.3.9.Linux.x86_64-gnu.cuda-12.9.tar.gz landrew@computelab-frontend-6:/home/scratch.jdaw_coreai/landrew/tensorrt/
# wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.13.3/tars/TensorRT-10.13.3.9.Linux.x86_64-gnu.cuda-12.9.tar.gz
tar -xvf TensorRT-10.*.tar.gz
export TENSORRT_ROOT=$HOME/tensorrt/TensorRT-10.13.3.9
export LD_LIBRARY_PATH=$TENSORRT_ROOT/lib:$LD_LIBRARY_PATH
export CMAKE_PREFIX_PATH=$TENSORRT_ROOT:$CMAKE_PREFIX_PATH

# ucx
cd $HOME
git clone https://github.com/openucx/ucx.git
cd ucx
git checkout v1.15.0
./autogen.sh
./configure --prefix=$HOME/ucx --enable-mt
make -j$(nproc)
make install
export UCX_CMAKE_DIR="$HOME/ucx/lib/cmake/ucx"

if ! grep -q 'include_guard' "$UCX_CMAKE_DIR/ucx-targets.cmake"; then
  sed -i '1i include_guard(GLOBAL)' "$UCX_CMAKE_DIR/ucx-targets.cmake"
fi

export ucx_DIR="$HOME/ucx/lib/cmake/ucx"
export LD_LIBRARY_PATH=$HOME/ucx/lib:$LD_LIBRARY_PATH

# cutlass_library
python -m pip install -e 3rdparty/cutlass

# zeromq
cd $HOME && git clone https://github.com/zeromq/libzmq.git
cd libzmq
git checkout v4.3.5
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=$HOME/zmq -DZMQ_BUILD_TESTS=OFF -DWITH_PERF_TOOL=OFF ..
cmake --build . -j
cmake --install .
export PKG_CONFIG_PATH="$HOME/zmq/lib/pkgconfig:$PKG_CONFIG_PATH"
export LD_LIBRARY_PATH="$HOME/zmq/lib:$LD_LIBRARY_PATH"
export LIBRARY_PATH="$HOME/zmq/lib:$LIBRARY_PATH"

cd $HOME/TensorRT-LLM
python scripts/build_wheel.py --clean
pip install ./build/tensorrt_llm-*.whl
```

setup commands:
```
export HOME=/home/scratch.jdaw_coreai/landrew
CMAKE_VER=3.30.4
ARCH=x86_64
export PATH="$HOME/opt/cmake-${CMAKE_VER}-linux-${ARCH}/bin:$PATH"

export CUDA_HOME=$HOME/cuda-12.9
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export NCCL_ROOT=$HOME/nccl/nccl_2.28.3-1+cuda13.0_x86_64
export CMAKE_PREFIX_PATH=$NCCL_ROOT
export CPATH=$NCCL_ROOT/include:$CPATH
export LD_LIBRARY_PATH=$NCCL_ROOT/lib:$LD_LIBRARY_PATH

export TENSORRT_ROOT=$HOME/tensorrt/TensorRT-10.13.3.9
export LD_LIBRARY_PATH=$TENSORRT_ROOT/lib:$LD_LIBRARY_PATH
export CMAKE_PREFIX_PATH=$TENSORRT_ROOT:$CMAKE_PREFIX_PATH

export UCX_CMAKE_DIR="$HOME/ucx/lib/cmake/ucx"
export ucx_DIR="$HOME/ucx/lib/cmake/ucx"
export LD_LIBRARY_PATH=$HOME/ucx/lib:$LD_LIBRARY_PATH

export CUDA_HOME=$HOME/cuda-12.9
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export PKG_CONFIG_PATH="$HOME/zmq/lib/pkgconfig:$PKG_CONFIG_PATH"
export LD_LIBRARY_PATH="$HOME/zmq/lib:$LD_LIBRARY_PATH"
export LIBRARY_PATH="$HOME/zmq/lib:$LIBRARY_PATH"
```

run commands:
```
crun -i -q "gpu.product_name=*H100_NVL* and cpu.arch=x86_64 and gpus=1 and gpu.memory_total_gb>40" -t 10:00:00
TORCH_CUDA_ARCH_LIST="9.0" mpirun -np 1 python bench.py
```
