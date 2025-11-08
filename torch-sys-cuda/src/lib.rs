/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/// A companion to the `torch-sys` crate that provides bindings for
/// CUDA-specific functionality from libtorch. This crate is separated out to
/// make it easier for clients who want to avoid compiling CUDA code.
///
/// The same safety logic described in the `torch-sys` crate applies here.
///
/// Note: This crate is CUDA-specific and is disabled when building for ROCm,
/// as ROCm-built PyTorch headers still reference CUDA headers which don't exist.

#[cfg(not(target_os = "macos"))]
#[cfg(not(rocm_disabled))]
mod bridge;

#[cfg(not(target_os = "macos"))]
#[cfg(not(rocm_disabled))]
pub mod cuda;

#[cfg(not(target_os = "macos"))]
#[cfg(not(rocm_disabled))]
pub mod nccl;

// For ROCm builds, use the HIP bridge implementation
#[cfg(rocm_disabled)]
mod bridge_rocm;

#[cfg(rocm_disabled)]
mod cuda_rocm;

#[cfg(rocm_disabled)]
mod nccl_rocm;

#[cfg(rocm_disabled)]
pub mod cuda {
    pub use super::cuda_rocm::*;
}

#[cfg(rocm_disabled)]
pub mod nccl {
    pub use super::nccl_rocm::*;
}

