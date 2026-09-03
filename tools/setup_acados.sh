#!/usr/bin/env bash
set -eo pipefail

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

case "$(uname -s)-$(uname -m)" in
  Linux-x86_64)
    LIB_EXT="so"
    PARALLEL_JOBS="$(nproc)"
    T_RENDERER_URL="https://github.com/acados/tera_renderer/releases/download/v0.0.34/t_renderer-v0.0.34-linux"
    CMAKE_PLATFORM_ARGS=()
    ;;
  Darwin-arm64)
    LIB_EXT="dylib"
    PARALLEL_JOBS="$(sysctl -n hw.ncpu)"
    T_RENDERER_URL="https://github.com/acados/tera_renderer/releases/download/v0.2.0/t_renderer-v0.2.0-osx-arm64"
    CMAKE_PLATFORM_ARGS=(-DBLASFEO_TARGET=ARMV8A_APPLE_M1)
    ;;
  *)
    echo "[Setup Acados] ERROR: Unsupported platform $(uname -s)-$(uname -m)."
    exit 1
    ;;
esac

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
if [ ! -f ${ACADOS_DIR}/lib/libacados.${LIB_EXT} ]; then
  echo "[Setup Acados] Building acados..."
  mkdir -p ${ACADOS_DIR}/build
  (
    cd ${ACADOS_DIR}/build
    cmake -DACADOS_WITH_QPOASES=ON -DCMAKE_POLICY_VERSION_MINIMUM=3.5 "${CMAKE_PLATFORM_ARGS[@]}" ..
    make install -j"$PARALLEL_JOBS"
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
  curl -L $T_RENDERER_URL \
    -o ${ACADOS_DIR}/bin/t_renderer
  chmod +x ${ACADOS_DIR}/bin/t_renderer
fi

if [ "${LIB_EXT}" = "dylib" ]; then
  # Avoid relying on DYLD_LIBRARY_PATH, which macOS may remove when launching
  # protected system processes. Resolve local dependencies via @loader_path.
  patch_dependency() {
    local library="$1"
    local old_path="$2"
    local new_path="$3"

    if /usr/bin/otool -L "${library}" | awk 'NR > 1 {print $1}' | grep -Fxq "${old_path}"; then
      /usr/bin/install_name_tool -change "${old_path}" "${new_path}" "${library}"
    fi
  }

  patch_dependency \
    "${ACADOS_DIR}/lib/libacados.dylib" \
    "libqpOASES_e.dylib" \
    "@loader_path/libqpOASES_e.dylib"
  patch_dependency \
    "${ACADOS_DIR}/lib/libacados.dylib" \
    "@rpath/libhpipm.dylib" \
    "@loader_path/libhpipm.dylib"
  patch_dependency \
    "${ACADOS_DIR}/lib/libacados.dylib" \
    "@rpath/libblasfeo.dylib" \
    "@loader_path/libblasfeo.dylib"
  patch_dependency \
    "${ACADOS_DIR}/lib/libhpipm.dylib" \
    "@rpath/libblasfeo.dylib" \
    "@loader_path/libblasfeo.dylib"

  for library in \
    "${ACADOS_DIR}/lib/libacados.dylib" \
    "${ACADOS_DIR}/lib/libhpipm.dylib" \
    "${ACADOS_DIR}/lib/libblasfeo.dylib" \
    "${ACADOS_DIR}/lib/libqpOASES_e.dylib"; do
    if ! /usr/bin/codesign --verify "${library}" >/dev/null 2>&1; then
      /usr/bin/codesign --force --sign - "${library}"
    fi
  done
fi

# Setting Environment Variables
if [ -f ${ACADOS_DIR}/lib/libacados.${LIB_EXT} ]; then
  export ACADOS_SOURCE_DIR="$ACADOS_DIR"
  export ACADOS_INSTALL_DIR="$ACADOS_DIR"
  if [ "${LIB_EXT}" = "so" ]; then
    export LD_LIBRARY_PATH="$ACADOS_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
  export PATH="${ACADOS_DIR}/interfaces/acados_template:${PATH}"
fi

echo "[Setup Acados] Acados is ready!"
