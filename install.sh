#!/bin/bash
#
# This script installs all dependencies (PyTorch, CUDA, C-libs)
# for your project on a bare aarch64 system with NO sudo.
#
# It uses Mambaforge and a two-stage process to resolve
# complex CUDA dependency conflicts.
#
# --- IMPORTANT ---
# Run this script from the root of your project directory
# (i.e., where your 'pyproject.toml' file is located).
#
set -e

# --- 1. Configuration ---
ENV_NAME="groot"
PYTHON_VERSION=3.10
# Pin ffmpeg to <5, as decord often requires v4.
FFMPEG_PIN="ffmpeg<5"
MAMBA_INSTALL_PATH="$HOME/miniforge3"
MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"

log() {
  echo -e "\n--- [INFO] $1 ---\n"
}

# --- 2. Install Mambaforge (if not present) ---
if [ -d "$MAMBA_INSTALL_PATH" ]; then
    log "Mambaforge is already installed at $MAMBA_INSTALL_PATH. Skipping installation."
else
    log "Mambaforge not found. Installing to $MAMBA_INSTALL_PATH..."
    curl -L -o miniforge_installer.sh $MINIFORGE_URL
    bash miniforge_installer.sh -b -p $MAMBA_INSTALL_PATH
    rm miniforge_installer.sh
    log "Mambaforge installation complete."
fi

# Initialize Mamba for this script session
source "$MAMBA_INSTALL_PATH/etc/profile.d/conda.sh"

# --- 3. Create Environment (Two-Stage Install) ---
log "Removing existing environment '$ENV_NAME' (if any)..."
$MAMBA_INSTALL_PATH/bin/mamba env remove -n $ENV_NAME -y || true

# 3a. STAGE 1: Install PyTorch, its CUDA toolkit, and C-Libs
log "Creating new Mamba environment (Stage 1: PyTorch + Libs)..."
$MAMBA_INSTALL_PATH/bin/mamba create -n $ENV_NAME -c pytorch -c conda-forge \
  python=$PYTHON_VERSION \
  pytorch \
  torchvision \
  torchaudio \
  c-compiler \
  cxx-compiler \
  make \
  cmake \
  nasm \
  git \
  $FFMPEG_PIN \
  xorg-libsm \
  xorg-libxext \
  hdf5 \
  tesseract \
  gtk3 \
  tbb \
  libgl \
  openblas \
  libjpeg-turbo \
  zlib \
  -y

log "Environment created. Activating '$ENV_NAME' for this script..."
conda activate $ENV_NAME

# 3b. STAGE 2: Install the CUDA Compiler
log "Installing CUDA Compiler (Stage 2)..."
$MAMBA_INSTALL_PATH/bin/mamba install -n $ENV_NAME -c conda-forge \
  cuda-compiler \
  -y

# --- 4. Install Python Dependencies (pip) ---
log "Setting PIP index and upgrading pip..."
export PIP_INDEX_URL=https://pypi.jetson-ai-lab.io/sbsa/cu130
export PIP_TRUSTED_HOST=pypi.jetson-ai.lab.io
pip install --upgrade pip setuptools

# 4a. Build PyTorch3D
log "Building and installing pytorch3d (main) from source..."
BUILD_DIR_3D=$(mktemp -d)
pushd $BUILD_DIR_3D
# Cleanup any old attempt
rm -rf pytorch3d/
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
pip install fvcore iopath

# --- THIS IS THE FIX for Pulsar build failures on aarch64 ---
log "Exporting BUILD_PYTORCH3D_WITH_PULSAR=0 to skip Pulsar build..."
export BUILD_PYTORCH3D_WITH_PULSAR=0
# -----------------------------------------------------------

$MAMBA_INSTALL_PATH/bin/mamba install -c conda-forge cudatoolkit
export LD_LIBRARY_PATH=/root/miniforge3/envs/groot/lib:$LD_LIBRARY_PATH

# Now install (no build isolation is key)
pip install . --no-build-isolation

popd
rm -rf $BUILD_DIR_3D

# 4b. Install your project
log "Checking for project file..."
if [ ! -f "pyproject.toml" ]; then
    echo "[ERROR] pyproject.toml not found."
    echo "Please run this script from the root of your project directory."
    exit 1
fi
log "Installing 'pyproject.toml' dependencies (with 'thor' extra)..."
pip install -e .[thor]

# --- 5. Build and Install Decord ---
log "Building and installing decord from source..."
BUILD_DIR_DECORD=$(mktemp -d)
pushd $BUILD_DIR_DECORD

git clone --recursive https://github.com/dmlc/decord
cd decord
mkdir build && cd build
# Point cmake to the conda env's ffmpeg and other libs
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$CONDA_PREFIX
make -j$(nproc)

log "Installing decord..."
cd ../python
pip install .

popd
rm -rf $BUILD_DIR_DECORD

# --- 6. Install Flash-Attn ---
log "Installing flash-attn (this will build)..."
# pip install flash-attn --no-build-isolation

log "🎉 Setup complete!"
echo "--- IMPORTANT NEXT STEPS ---"
echo "1. Close and RE-OPEN your terminal, or run:"
echo "   source $MAMBA_INSTALL_PATH/etc/profile.d/conda.sh"
echo "2. Activate your new environment:"
echo "   mamba activate $ENV_NAME"
