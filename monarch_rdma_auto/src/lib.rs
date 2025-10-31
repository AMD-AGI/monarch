/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! # Monarch RDMA Auto
//!
//! This crate automatically selects the appropriate RDMA backend based on the
//! `USE_ROCM` environment variable:
//!
//! - When `USE_ROCM=0` or unset: Uses CUDA backend (`monarch_rdma`)
//! - When `USE_ROCM=1`: Uses ROCm/HIP backend (`monarch_rdma_hip`)
//!
//! ## Usage
//!
//! ```bash
//! # Use CUDA (default)
//! cargo build -p monarch_rdma_auto
//!
//! # Use ROCm
//! USE_ROCM=1 ROCM_PATH=/path/to/rocm cargo build -p monarch_rdma_auto --features rocm --no-default-features
//! ```
//!
//! ## Re-exports
//!
//! This crate re-exports all public items from the selected backend, so you can use it
//! as a drop-in replacement:
//!
//! ```ignore
//! use monarch_rdma_auto::*;
//! ```

// Re-export everything from the selected backend
#[cfg(feature = "cuda")]
pub use monarch_rdma::*;

#[cfg(feature = "rocm")]
pub use monarch_rdma_hip::*;

// Compile-time check to ensure exactly one backend is selected
#[cfg(not(any(feature = "cuda", feature = "rocm")))]
compile_error!("Either 'cuda' or 'rocm' feature must be enabled");

#[cfg(all(feature = "cuda", feature = "rocm"))]
compile_error!("Cannot enable both 'cuda' and 'rocm' features at the same time");
