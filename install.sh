#!/bin/bash
#
# This script installs Mambaforge (if not present) and then mimics the
# Dockerfile setup on a native aarch64 system.
#
# It requires NO sudo privileges.
#
# Requirements:
# 1. Run on aarch64 (Ubuntu-based).
# 2. 'pyproject.toml' must exist in the current directory.
#

# Exit immediately if a command exits with a non-zero status
set -e

# --- Configuration ---
ENV_NAME="groot"
PYTHON_VERSION=3.10
FFMPEG_VERSION="4.4.2" 
MAMBA_INSTALL_PATH="$HOME/miniforge3"
MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"

# --- Helper Function ---
log() {
  echo -e "\n[INFO] $1\n"
}

# --- Main Functions ---

check_prereqs() {
  log "Checking prerequisites..."
  if [ "$(uname -m)" != "aarch64" ]; then
    echo "[ERROR] This script is intended for aarch64 systems."
    exit 1
  fi

  if [ ! -f "pyproject.toml" ]; then
    echo "[ERROR] 'pyproject.toml' not found."
    echo "Please run this script from your project's root directory."
    exit 1
  fi
  echo "Prerequisites met."
}

install_mambaforge() {
  if [ -d "$MAMBA_INSTALL_PATH" ]; then
    log "Mambaforge is already installed at $MAMBA_INSTALL_PATH. Skipping installation."
  else
    log "Mambaforge not found. Installing to $MAMBA_INSTALL_PATH..."
    log "Downloading Mambaforge installer..."
    # Use curl, follow redirects (-L), output to a file (-o)
    curl -L -o miniforge_installer.sh $MINIFORGE_URL

    log "Running Mambaforge installer in batch mode..."
    # -b: Batch mode (no prompts)
    # -p: Installation prefix (path)
    bash miniforge_installer.sh -b -p $MAMBA_INSTALL_PATH

    log "Cleaning up installer..."
    rm miniforge_installer.sh
    log "Mambaforge installation complete."
  fi

  log "Initializing Mamba for this script session..."
  # We MUST source this to make 'conda' and 'mamba' commands available
  # to the rest of this script.
  source "$MAMBA_INSTALL_PATH/etc/profile.d/conda.sh"
}

create_mamba_env() {
  log "Checking for existing environment '$ENV_NAME'..."
  # Deactivate if active, then remove. '|| true' ignores errors if env doesn't exist.
  $MAMBA_INSTALL_PATH/bin/conda deactivate || true
  $MAMBA_INSTALL_PATH/bin/mamba env remove -n $ENV_NAME -y || true
  
  log "Creating new Mamba environment '$ENV_NAME' with dependencies..."
  # This command replaces the entire 'apt-get install' block.
  # We get all build tools and C-libraries from conda-forge.
  $MAMBA_INSTALL_PATH/bin/mamba create -n $ENV_NAME -c conda-forge \
    python=$PYTHON_VERSION \
    pytorch \
    torchvision \
    torchaudio \
     cudatoolkit \
    c-compiler \
    cxx-compiler \
    make \
    cmake \
    nasm \
    git \
    xorg-libsm \
    xorg-libxext \
    ffmpeg=$FFMPEG_VERSION \
    hdf5 \
    tesseract \
    gtk3 \
    tbb \
    libgl \
    openblas \
    -y

  log "Environment created. Activating '$ENV_NAME'..."
  # Activate the new environment for the subsequent steps
  conda activate $ENV_NAME
}

install_python_deps() {
  # This function assumes 'conda activate $ENV_NAME' has just been run

  log "Setting Jetson-specific PIP index..."
  export PIP_INDEX_URL=https://pypi.jetson-ai-lab.io/sbsa/cu130
  export PIP_TRUSTED_HOST=pypi.jetson-ai-lab.io

  log "Upgrading pip and setuptools..."
  pip install --upgrade pip setuptools

  log "Installing project dependencies from 'pyproject.toml' (with 'thor' extra)..."
  pip install -e .[thor]
}

build_decord_from_source() {
  # This function assumes the conda env is active

  log "Building and installing 'decord' from source..."
  # We don't need to build ffmpeg; Mamba already installed it.
  # We just need to build decord against the ffmpeg in our env.

  BUILD_DIR=$(mktemp -d)
  pushd $BUILD_DIR

  log "Cloning decord..."
  git clone --recursive https://github.com/dmlc/decord
  cd decord
  mkdir -p build && cd build

  log "Configuring decord (cmake)..."
  # This tells cmake to find all dependencies (like ffmpeg)
  # inside our active Conda environment ($CONDA_PREFIX).
  cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH=$CONDA_PREFIX \
    -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX

  log "Building decord (this may take a while)..."
  make -j$(nproc)

  log "Installing decord into the Mamba environment..."
  cd ../python
  # This will install it into the active env's site-packages
  pip install .

  # Go back to original directory and clean up
  popd
  log "Cleaning up temporary build directory..."
  rm -rf $BUILD_DIR
}

# --- Main Execution ---
main() {
  check_prereqs
  install_mambaforge
  create_mamba_env
  install_python_deps
  build_decord_from_source

  log "🎉 Installation complete!"
  echo
  echo "Mambaforge is installed in: $MAMBA_INSTALL_PATH"
  echo "Your project environment '$ENV_NAME' is ready."
  echo
  echo "--- IMPORTANT NEXT STEPS ---"
  echo
  echo "1. Close and RE-OPEN your terminal, or run:"
  echo "   source ~/.bashrc"
  echo
  echo "2. Activate your new environment by running:"
  echo "   mamba activate $ENV_NAME"
  echo
  echo "(No need to set LD_LIBRARY_PATH; Mamba handles it automatically.)"
}

# Run the main function
main
