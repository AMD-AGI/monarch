#!/usr/bin/env bash

set -euo pipefail

# Core locations and versions (override via environment variables if needed).
MINIFORGE_DIR=${MINIFORGE_DIR:-"$HOME/miniforge3"}
ENV_NAME=${ENV_NAME:-monarch-env}
PYTHON_VERSION=${PYTHON_VERSION:-3.12}
WORKSPACE=${WORKSPACE:-"$HOME/monarch-workspace"}
TORCHTITAN_REF=${TORCHTITAN_REF:-61c25f8d3bf1792f6c4b80417b9a1f5dd464deaf}
MONARCH_REF=${MONARCH_REF:-xinyu/rdma}
TORCH_PACKAGES=${TORCH_PACKAGES:-"torch==2.9.0+rocm6.4 torchvision torchaudio"}
TORCH_INDEX_URL=${TORCH_INDEX_URL:-https://download.pytorch.org/whl/rocm6.4}
DEFAULT_USER=${SUDO_USER:-$(whoami)}
RENDER_USER=${RENDER_USER:-$DEFAULT_USER}
HF_TOKEN=${HF_TOKEN:-hf_your_token} # export HF_TOKEN=... before running to download assets
LLAMA_REPO_ID=${LLAMA_REPO_ID:-meta-llama/Llama-3.1-8B}
LLAMA_ASSETS=${LLAMA_ASSETS:-tokenizer}

log() {
  printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

die() {
  printf '\nERROR: %s\n' "$*" >&2
  exit 1
}

run_in_dir() {
  local dir=$1
  shift
  log "($dir) $*"
  (cd "$dir" && "$@")
}

ensure_miniforge() {
  if [ -d "$MINIFORGE_DIR" ]; then
    log "Found Miniforge at $MINIFORGE_DIR"
    return
  fi

  if ! command -v curl >/dev/null 2>&1; then
    if command -v apt-get >/dev/null 2>&1; then
      if ! command -v sudo >/dev/null 2>&1; then
        die "curl is missing and sudo is unavailable to install it."
      fi
      log "curl not found; installing via apt-get (sudo may prompt)"
      sudo apt-get update
      sudo apt-get install -y curl
    else
      die "curl is required to bootstrap Miniforge but automatic installation is unsupported on this system."
    fi
  fi

  local installer_name
  installer_name="Miniforge3-$(uname)-$(uname -m).sh"
  local installer_url
  installer_url="https://github.com/conda-forge/miniforge/releases/latest/download/${installer_name}"
  local installer_path="/tmp/${installer_name}"

  log "Miniforge not found; downloading $installer_url"
  curl -L -o "$installer_path" "$installer_url"
  chmod +x "$installer_path"

  log "Installing Miniforge into $MINIFORGE_DIR"
  bash "$installer_path" -b -p "$MINIFORGE_DIR"
  rm -f "$installer_path"

  if [ -f "$HOME/.bashrc" ]; then
    # shellcheck disable=SC1090
    source "$HOME/.bashrc"
  fi
}

ensure_libclang() {
  if command -v clang >/dev/null 2>&1 \
    && command -v ldconfig >/dev/null 2>&1 \
    && ldconfig -p 2>/dev/null | grep -q 'libclang\\.so'; then
    log "Found existing clang and libclang"
    return
  fi

  if ! command -v apt-get >/dev/null 2>&1; then
    die "apt-get not found; install clang and libclang-dev manually"
  fi

  if ! command -v sudo >/dev/null 2>&1; then
    die "sudo is required to install clang and libclang-dev"
  fi

  log "Installing clang and libclang-dev via apt-get (sudo may prompt)"
  sudo apt-get update
  sudo apt-get install -y clang libclang-dev
}

ensure_render_group() {
  local user=$RENDER_USER
  if ! id "$user" >/dev/null 2>&1; then
    die "User $user not found; set RENDER_USER=<name> if needed."
  fi

  local groups
  groups=$(id -nG "$user")
  local needs_change=0

  for group in render video; do
    if ! getent group "$group" >/dev/null 2>&1; then
      die "Required group '$group' does not exist on this system."
    fi
    if ! grep -qw "$group" <<<"$groups"; then
      needs_change=1
    fi
  done

  if [ "$needs_change" -eq 0 ]; then
    log "$user already belongs to render and video groups"
    return
  fi

  if ! command -v sudo >/dev/null 2>&1; then
    die "sudo is required to modify group membership for $user"
  fi

  log "Adding $user to render and video groups (sudo may prompt)"
  sudo usermod -aG render,video "$user"

  log "Spawning short-lived newgrp render shell to refresh membership"
  newgrp render <<'EOF'
echo "render group shell refreshed; exiting."
EOF
  log "Group changes recorded; open a new shell or run 'newgrp render' manually if needed."
}

activate_conda() {
  if [ -f "$MINIFORGE_DIR/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1090
    source "$MINIFORGE_DIR/etc/profile.d/conda.sh"
  else
    eval "$("$MINIFORGE_DIR/bin/conda" shell.bash hook)"
  fi

  if ! conda info --envs | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    log "Creating conda environment $ENV_NAME (python=$PYTHON_VERSION)"
    conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
  else
    log "Found existing conda environment $ENV_NAME"
  fi

  log "Activating conda environment $ENV_NAME"
  conda activate "$ENV_NAME"
}

install_base_packages() {
  log "Installing libunwind via conda"
  conda install -y libunwind

  log "Upgrading pip/setuptools/wheel"
  python -m pip install -U pip setuptools wheel

  log "Installing uv"
  python -m pip install -U uv
}

setup_rust() {
  if ! command -v rustup >/dev/null 2>&1; then
    log "Installing rustup"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
  else
    log "rustup already installed"
  fi

  if [ -f "$HOME/.cargo/env" ]; then
    # shellcheck disable=SC1090
    source "$HOME/.cargo/env"
  fi

  log "Ensuring nightly toolchain"
  rustup toolchain install nightly
  rustup default nightly
}

install_torch_stack() {
  log "Installing ROCm PyTorch stack via uv"
  if [ -z "${TORCH_PACKAGES// }" ]; then
    die "TORCH_PACKAGES is empty; set it to a space separated package list"
  fi

  # shellcheck disable=SC2206 
  local torch_pkgs=( ${TORCH_PACKAGES} )
  uv pip install "${torch_pkgs[@]}" \
    --index-url "$TORCH_INDEX_URL" \
    --force-reinstall

  log "Adjusting ROCm shared library symlinks inside PyTorch"
  local torch_lib_dir
  torch_lib_dir=$(python - <<'PY'
import os
import torch
print(os.path.join(os.path.dirname(torch.__file__), "lib"))
PY
)

  local -a pairs=(
    "libamdhip64.so.6 libamdhip64.so"
    "libhsa-runtime64.so.1 libhsa-runtime64.so"
    "librccl.so.1 librccl.so"
    "librocprofiler-register.so.0 librocprofiler-register.so"
    "librocm_smi64.so.7 librocm_smi64.so"
    "libdrm.so.2 libdrm.so"
    "libdrm_amdgpu.so.1 libdrm_amdgpu.so"
  )

  local pair
  for pair in "${pairs[@]}"; do
    IFS=' ' read -r target link <<<"$pair"
    rm -f "$torch_lib_dir/$target"
    ln -s "$link" "$torch_lib_dir/$target"
  done
}

ensure_repo() {
  local repo_url=$1
  local dest=$2
  local ref=$3

  if [ ! -d "$dest/.git" ]; then
    log "Cloning $repo_url into $dest"
    git clone "$repo_url" "$dest"
  else
    log "Reusing existing repo at $dest"
  fi

  run_in_dir "$dest" git fetch origin
  run_in_dir "$dest" git fetch origin --tags
  run_in_dir "$dest" git checkout "$ref"

  local branch
  branch="HEAD"
  local branch_name
  if branch_name=$(cd "$dest" && git rev-parse --abbrev-ref HEAD 2>/dev/null); then
    branch="$branch_name"
  fi
  if [ "$branch" != "HEAD" ]; then
    run_in_dir "$dest" git pull --ff-only origin "$branch"
  fi
}

setup_torchtitan() {
  local dest="$WORKSPACE/torchtitan"
  ensure_repo https://github.com/pytorch/torchtitan.git "$dest" "$TORCHTITAN_REF"

  run_in_dir "$dest" python -m pip install -r requirements.txt
  run_in_dir "$dest" python -m pip install -e .
}

setup_monarch() {
  local dest="$WORKSPACE/monarch"
  ensure_repo https://github.com/AMD-AGI/monarch.git "$dest" "$MONARCH_REF"

  run_in_dir "$dest" uv pip install -r build-requirements.txt
  if ! ulimit -n 2048; then
    log "Unable to raise open file limit to 2048, continuing anyway"
  fi

  local library_path="${CONDA_PREFIX:-}/lib"
  if [ -z "${CONDA_PREFIX:-}" ] || [ ! -d "$library_path" ]; then
    die "Expected active conda env with libraries under $library_path"
  fi

  run_in_dir "$dest" env LIBRARY_PATH="$library_path" uv pip install --no-build-isolation -e .
}

download_llama_assets() {
  local dest="$WORKSPACE/torchtitan"
  if [ ! -d "$dest" ]; then
    die "torchtitan repository not found at $dest; run setup_torchtitan first"
  fi

  if [ -z "${HF_TOKEN:-}" ] || [ "$HF_TOKEN" = "hf_your_token" ]; then
    die "Set HF_TOKEN to a valid Hugging Face token to download the dataset assets"
  fi

  if [ -z "${LLAMA_ASSETS// }" ]; then
    die "LLAMA_ASSETS is empty; set it to a space separated list (e.g. 'tokenizer weights')"
  fi

  # shellcheck disable=SC2206 
  local asset_list=( ${LLAMA_ASSETS} )
  run_in_dir "$dest" python ./scripts/download_hf_assets.py \
    --repo_id "$LLAMA_REPO_ID" \
    --assets "${asset_list[@]}" \
    --hf_token "$HF_TOKEN"
}

main() {
  log "Workspace root: $WORKSPACE"
  ensure_miniforge
  ensure_render_group
  ensure_libclang
  mkdir -p "$WORKSPACE"
  cd "$WORKSPACE"

  activate_conda
  install_base_packages
  setup_rust
  install_torch_stack
  setup_monarch
  setup_torchtitan
  download_llama_assets

  log "Monarch + Torchtitan setup complete!"
}

main "$@"
