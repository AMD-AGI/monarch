/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/hip/HIPEvent.h>
#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>
#include <rccl/rccl.h>
#include <torch/torch.h>

namespace monarch {

// HIP Event APIs (uses at::cuda namespace for compatibility)
std::unique_ptr<at::cuda::CUDAEvent>
create_hip_event(bool enable_timing, bool blocking, bool interprocess);

// Helper to convert HIPStream to HIPStreamMasqueradingAsCUDA for event operations
void record_event(at::cuda::CUDAEvent& event, const c10::hip::HIPStream& stream);
void block_event(at::cuda::CUDAEvent& event, const c10::hip::HIPStream& stream);

// HIP Stream APIs
std::shared_ptr<c10::hip::HIPStream> get_current_hip_stream(
    c10::DeviceIndex device);

std::shared_ptr<c10::hip::HIPStream> create_hip_stream(
    c10::DeviceIndex device,
    int32_t priority);

void set_current_hip_stream(const c10::hip::HIPStream& stream);

// Get raw HIP stream handle as usize
size_t get_stream_handle(const c10::hip::HIPStream& stream);

/// This function exists because ncclConfig initialization requires the use of
/// a macro. We cannot reference the macro directly from Rust code, so we wrap
/// the macro use in a function and bind that to Rust instead.
inline ncclConfig_t make_nccl_config() {
  ncclConfig_t ret = NCCL_CONFIG_INITIALIZER;
  return ret;
}

} // namespace monarch
