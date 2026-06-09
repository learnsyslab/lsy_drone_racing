#!/usr/bin/env bash
set -euo pipefail

# Check if environment variable is already set
if [ -z "$PIXI_PROJECT_ROOT" ]; then
    echo "[Setup Acados] Not running inside a Pixi environment; skipping setup_acados.sh"
    exit 0
fi

# Check if pixi env is properly set up
if [ ! -f ${PIXI_PROJECT_ROOT}/pixi.lock ]; then
  echo "[Setup Acados] ERROR: pixi environment is not properly set up."
  exit 0
fi

ACADOS_DIR="${PIXI_PROJECT_ROOT}/acados"

# Clone and build acados
if [ ! -d ${ACADOS_DIR}/.git ]; then
  echo "[Setup Acados] Cloning acados..."
  git clone https://github.com/acados/acados.git ${ACADOS_DIR}
  (
    cd ${ACADOS_DIR}
    git checkout tags/v0.5.1
    git submodule update --recursive --init
  )
fi

# Check if pip is installed
if ! command -v pip >/dev/null 2>&1; then
  echo "[Setup Acados] ERROR: pip is not installed. Please install pip first."
  exit 0
fi

# Build Acados
if [ ! -f ${ACADOS_DIR}/lib/libacados.so ]; then
  echo "[Setup Acados] Building acados..."
  mkdir -p ${ACADOS_DIR}/build
  (
    cd ${ACADOS_DIR}/build
    cmake -DACADOS_WITH_QPOASES=ON -DCMAKE_POLICY_VERSION_MINIMUM=3.5 ..
    make install -j"$(nproc)"
  )
fi

# Install Acados Python interface
if ! pip show acados-template >/dev/null 2>&1; then
  echo "[Setup Acados] Installing acados Python interface..."
  pip install -e ${ACADOS_DIR}/interfaces/acados_template
fi

# Download Tera Renderer
if [ ! -f ${ACADOS_DIR}/bin/t_renderer ]; then
  echo "[Setup Acados] Downloading tera_renderer..."
  mkdir -p ${ACADOS_DIR}/bin
  curl -L https://github.com/acados/tera_renderer/releases/download/v0.0.34/t_renderer-v0.0.34-linux \
    -o ${ACADOS_DIR}/bin/t_renderer
  chmod +x ${ACADOS_DIR}/bin/t_renderer
fi

# Setting Environment Variables
if [ -f ${ACADOS_DIR}/lib/libacados.so ]; then
  export ACADOS_SOURCE_DIR="$ACADOS_DIR"
  export ACADOS_INSTALL_DIR="$ACADOS_DIR"
  export LD_LIBRARY_PATH="$ACADOS_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export PATH="${ACADOS_DIR}/interfaces/acados_template:${PATH}"
fi

echo "[Setup Acados] Acados is ready!"
