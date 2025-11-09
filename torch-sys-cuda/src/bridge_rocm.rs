/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#[allow(dead_code)]
#[cxx::bridge(namespace = "monarch")]
pub(crate) mod ffi {
    unsafe extern "C++" {
        include!("ATen/hip/HIPEvent.h");
        include!("monarch/torch-sys-cuda/src/bridge_rocm.h");

        // HIP Event APIs (uses at::cuda namespace for compatibility)
        #[namespace = "at::cuda"]
        type CUDAEvent;
        fn create_hip_event(
            enable_timing: bool,
            blocking: bool,
            interprocess: bool,
        ) -> UniquePtr<CUDAEvent>;
        fn record_event(event: Pin<&mut CUDAEvent>, stream: &HIPStream);
        fn block_event(event: Pin<&mut CUDAEvent>, stream: &HIPStream);
        fn query(self: &CUDAEvent) -> bool;
        fn elapsed_time(self: &CUDAEvent, end_event: &CUDAEvent) -> f32;
        fn synchronize(self: &CUDAEvent);

        // HIP Stream APIs
        #[namespace = "c10::hip"]
        type HIPStream;
        fn get_current_hip_stream(device: i8) -> SharedPtr<HIPStream>;
        fn set_current_hip_stream(stream: &HIPStream);
        fn create_hip_stream(device: i8, priority: i32) -> SharedPtr<HIPStream>;
        fn query(self: &HIPStream) -> bool;
        fn synchronize(self: &HIPStream);
        fn device_index(self: &HIPStream) -> i8;
        // Returns the raw HIP stream handle as usize
        fn get_stream_handle(stream: &HIPStream) -> usize;

        // RCCL helpers
        #[namespace = ""]
        type ncclConfig_t = nccl_sys::ncclConfig_t;
        fn make_nccl_config() -> ncclConfig_t;
    }
}

use std::fmt::Debug;
use std::fmt::Error;
use std::fmt::Formatter;

// SAFETY: HIPStream is thread safe
unsafe impl Send for ffi::HIPStream {}
// SAFETY: see above
unsafe impl Sync for ffi::HIPStream {}

impl Debug for ffi::HIPStream {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        f.debug_struct("HIPStream")
            .field("device", &format!("{}", self.device_index()))
            .field("stream", &format!("{:p}", self))
            .finish()
    }
}

impl Debug for ffi::CUDAEvent {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        f.debug_struct("CUDAEvent")
            .field("ptr", &format!("{:p}", self))
            .finish()
    }
}

// SAFETY: CUDAEvent is thread safe
unsafe impl Send for ffi::CUDAEvent {}
// SAFETY: see above
unsafe impl Sync for ffi::CUDAEvent {}
