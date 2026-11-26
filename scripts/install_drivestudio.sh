#!/bin/bash
# drivestudio official install (Python 3.9 / torch 2.0.0+cu117), adapted for WSL Ubuntu 24.04 + RTX 4060:
#   - CUDA 11.7 toolkit and gcc 11 come from conda (system gcc 13 is too new for nvcc 11.7)
#   - TORCH_CUDA_ARCH_LIST=8.6 (nvcc 11.7 does not know sm_89; sm_86 cubins run on sm_89)
set -eo pipefail
STEP() { echo; echo "=== [$(date +%H:%M:%S)] $*"; }
source ${CONDA_SH:-~/miniforge3/etc/profile.d/conda.sh}
DS=${DS:-~/gs/drivestudio}   # drivestudio checkout
cd $DS

if ! conda env list | grep -q "^drivestudio "; then
  STEP "conda create python 3.9"
  conda create -n drivestudio python=3.9 -y
fi
conda activate drivestudio

if ! command -v nvcc >/dev/null; then
  STEP "CUDA 11.7 toolkit + gcc/g++ 11 (conda)"
  conda install -y -c "nvidia/label/cuda-11.7.1" cuda-toolkit
  conda install -y -c conda-forge "gcc_linux-64=11" "gxx_linux-64=11"
fi
mkdir -p ~/gs/ccbin
ln -sf "$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc" ~/gs/ccbin/gcc
ln -sf "$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++" ~/gs/ccbin/g++
export PATH=~/gs/ccbin:$PATH CC=~/gs/ccbin/gcc CXX=~/gs/ccbin/g++
export CUDA_HOME=$CONDA_PREFIX TORCH_CUDA_ARCH_LIST="8.6" MAX_JOBS=1
nvcc --version | tail -2; gcc --version | head -1

STEP "chumpy (its setup.py imports pip, so it cannot build in an isolated env)"
pip install numpy==1.23.1 && pip install --no-build-isolation chumpy

STEP "pip install -r requirements.txt"
pip install -r requirements.txt

STEP "torch check"
python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

STEP "gsplat v1.3.0 (build)"
pip install --no-build-isolation "git+https://github.com/nerfstudio-project/gsplat.git@v1.3.0"

STEP "pytorch3d (build)"
pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git"

STEP "nvdiffrast"
pip install --no-build-isolation "git+https://github.com/NVlabs/nvdiffrast"

STEP "smplx (editable)"
cd third_party/smplx && pip install -e . && cd ../..

STEP "import check"
python -c "import gsplat, pytorch3d, nvdiffrast.torch, smplx; print('gsplat', gsplat.__version__, 'pytorch3d', pytorch3d.__version__)"
STEP "DONE"
