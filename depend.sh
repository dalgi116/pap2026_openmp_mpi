#!/usr/bin/env bash
set -euo pipefail

main() {
  # Run from repo root and use sudo only when not already root.
  cd "$(dirname "${BASH_SOURCE[0]}")"
  local SUDO=""
  [ "$(id -u)" -ne 0 ] && SUDO="sudo"

  # Install compiler (with OpenMP), pkg-config, OpenMPI, and OpenCV
  $SUDO apt-get update
  $SUDO apt-get install -y \
    build-essential \
    pkg-config \
    openmpi-bin \
    libopenmpi-dev \
    libopencv-dev

  # Ensure build and test directories exist, and ignore build artifacts in Git
  mkdir -p build test/out
  printf '*\n' > build/.gitignore
  printf '*\n' > test/out/.gitignore

  echo "Environment setup complete:"
  echo "  mpiCC:  $(command -v mpiCC)"
  echo "  mpirun: $(command -v mpirun)"
  echo "  opencv: $(pkg-config --modversion opencv4)"
}

main "$@"
