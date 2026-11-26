#!/bin/bash
# SegFormer env for drivestudio's sky masks (docs/NuScenes.md section 4), Python 3.8 + torch 1.8.1+cu111 + mmcv-full 1.2.7.
#   - mmcv-full 1.2.7 has no prebuilt wheel for torch 1.8 -> built from source as the docs say; CUDA is hidden during the
#     build so only its CPU ops compile (no CUDA 11.1 compiler needed; SegFormer inference uses no mmcv CUDA op).
#   - gcc 9 from conda (torch 1.8 extension headers predate gcc 13); numpy < 1.24 (mmcv 1.2.7 uses np.float etc.)
set -eo pipefail
STEP() { echo; echo "=== [$(date +%H:%M:%S)] $*"; }
SEGFORMER=${SEGFORMER:-~/gs/SegFormer}
source ${CONDA_SH:-~/miniforge3/etc/profile.d/conda.sh}
if ! conda env list | grep -q "^segformer "; then
  STEP "conda create python 3.8 + gcc 9"
  conda create -n segformer python=3.8 -y
  conda install -n segformer -y -c conda-forge "gcc_linux-64=9" "gxx_linux-64=9"
fi
conda activate segformer
STEP "torch 1.8.1+cu111"
python -c "import torch" 2>/dev/null || pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
STEP "other deps"
pip install "numpy<1.24" timm==0.3.2 pylint debugpy opencv-python-headless attrs ipython tqdm imageio scikit-image omegaconf gdown
STEP "mmcv-full 1.2.7 (source build, CPU ops)"
# conda gcc sysroot has no crypt.h (included by Python 3.8 headers) -> libxcrypt from conda-forge
conda list libxcrypt | grep -q libxcrypt || conda install -y -c conda-forge libxcrypt
export CPATH=$CONDA_PREFIX/include
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
CUDA_VISIBLE_DEVICES="" MMCV_WITH_OPS=1 MAX_JOBS=1 pip install --no-build-isolation --no-cache-dir mmcv-full==1.2.7
STEP "SegFormer code"
[ -d $SEGFORMER ] || git clone https://github.com/NVlabs/SegFormer $SEGFORMER
cd $SEGFORMER && pip install --no-build-isolation .
STEP "SegFormer-B5 Cityscapes checkpoint (backup link from drivestudio docs)"
mkdir -p $SEGFORMER/pretrained
[ -s $SEGFORMER/pretrained/segformer.b5.1024x1024.city.160k.pth ] || gdown 1e7DECAH0TRtPZM6hTqRGoboq1XPqSmuj -O $SEGFORMER/pretrained/segformer.b5.1024x1024.city.160k.pth
ls -la $SEGFORMER/pretrained
STEP "import check"
cd ~ && python -c "import torch, mmcv, mmseg; from mmcv.ops import point_sample; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), 'mmcv', mmcv.__version__, 'mmseg', mmseg.__version__)"
STEP "DONE"
