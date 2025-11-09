/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! HIP stream and event bindings for ROCm using PyTorch's C++ HIP APIs

use std::time::Duration;

use cxx::SharedPtr;
use cxx::UniquePtr;
use derive_more::Into;
use cuda_sys::hipError_t;
use cuda_sys::hipSetDevice;
use cuda_sys::hipStream_t;
use thiserror::Error;
use torch_sys::CudaDevice;

use crate::bridge_rocm::ffi::{self};

/// Wrapper around a HIP stream.
#[derive(Debug, Clone, Into)]
#[into(ref)]
pub struct Stream {
    inner: SharedPtr<ffi::HIPStream>,
}

// SAFETY: HIPStream is thread safe
unsafe impl Send for Stream {}
// SAFETY: see above
unsafe impl Sync for Stream {}

impl Stream {
    /// Create a new stream on the current device, at priority 0.
    pub fn new() -> Self {
        Self {
            inner: ffi::create_hip_stream(-1, 0),
        }
    }

    /// Create a new stream on the specified device, at priority 0.
    pub fn new_with_device(device: CudaDevice) -> Self {
        Self {
            inner: ffi::create_hip_stream(device.index().into(), 0),
        }
    }

    /// Get the current stream on the current device.
    pub fn get_current_stream() -> Self {
        Self {
            inner: ffi::get_current_hip_stream(-1),
        }
    }

    /// Get the current stream on the specified device.
    pub fn get_current_stream_on_device(device: CudaDevice) -> Self {
        Self {
            inner: ffi::get_current_hip_stream(device.index().into()),
        }
    }

    /// Set the provided stream as the current stream. Also sets the current
    /// device to be the same as the stream's device.
    pub fn set_current_stream(stream: &Stream) {
        ffi::set_current_hip_stream(stream.as_ref())
    }

    /// Make all future work submitted to this stream wait for an event.
    pub fn wait_event(&self, event: &mut Event) {
        event.wait(Some(self))
    }

    /// Synchronize with another stream.
    pub fn wait_stream(&self, stream: &Stream) {
        self.wait_event(&mut stream.record_event(None))
    }

    /// Record an event on this stream.
    pub fn record_event(&self, event: Option<Event>) -> Event {
        let mut event = event.unwrap_or(Event::new());
        event.record(Some(self));
        event
    }

    /// Check if all work submitted to this stream has completed.
    pub fn query(&self) -> bool {
        self.inner.query()
    }

    /// Wait for all kernels in this stream to complete.
    pub fn synchronize(&self) {
        self.inner.synchronize()
    }

    pub fn stream(&self) -> hipStream_t {
        ffi::get_stream_handle(self.as_ref()) as hipStream_t
    }
}

impl AsRef<ffi::HIPStream> for Stream {
    fn as_ref(&self) -> &ffi::HIPStream {
        self.inner.as_ref().unwrap()
    }
}

impl PartialEq for Stream {
    fn eq(&self, other: &Self) -> bool {
        self.stream() == other.stream()
    }
}

/// Wrapper around a HIP event (uses CUDA namespace for compatibility).
#[derive(Debug)]
pub struct Event {
    inner: UniquePtr<ffi::CUDAEvent>,
}

impl Event {
    /// Create a new event.
    pub fn new() -> Self {
        Self {
            inner: ffi::create_hip_event(false, false, false),
        }
    }

    /// Record the event on the current stream.
    pub fn record(&mut self, stream: Option<&Stream>) {
        match stream {
            Some(stream) => ffi::record_event(self.inner.pin_mut(), stream.as_ref()),
            None => ffi::record_event(
                self.inner.pin_mut(),
                Stream::get_current_stream().as_ref(),
            ),
        }
    }

    /// Make all future work submitted to the given stream wait for this event.
    pub fn wait(&mut self, stream: Option<&Stream>) {
        match stream {
            Some(stream) => ffi::block_event(self.inner.pin_mut(), stream.as_ref()),
            None => ffi::block_event(
                self.inner.pin_mut(),
                Stream::get_current_stream().as_ref(),
            ),
        }
    }

    /// Check if all work currently captured by event has completed.
    pub fn query(&self) -> bool {
        self.inner.query()
    }

    /// Return the time elapsed.
    pub fn elapsed_time(&self, end_event: &Event) -> Duration {
        Duration::from_millis(self.inner.elapsed_time(end_event.as_ref()) as u64)
    }

    /// Wait for the event to complete.
    pub fn synchronize(&self) {
        self.inner.synchronize()
    }
}

impl AsRef<ffi::CUDAEvent> for Event {
    fn as_ref(&self) -> &ffi::CUDAEvent {
        self.inner.as_ref().unwrap()
    }
}

/// Corresponds to the HIP error codes.
#[derive(Debug, Error)]
pub enum CudaError {
    #[error("invalid value")]
    InvalidValue,
    #[error("memory allocation failed")]
    MemoryAllocation,
    #[error("initialization error")]
    InitializationError,
    #[error("no device")]
    NoDevice,
    #[error("invalid device")]
    InvalidDevice,
    #[error("unknown error")]
    Unknown,
}

pub fn cuda_check(result: hipError_t) -> Result<(), CudaError> {
    match result.0 {
        0 => Ok(()),
        1 => Err(CudaError::InvalidValue),
        2 => Err(CudaError::MemoryAllocation),
        3 => Err(CudaError::InitializationError),
        100 => Err(CudaError::NoDevice),
        101 => Err(CudaError::InvalidDevice),
        _ => Err(CudaError::Unknown),
    }
}

pub fn set_device(device: CudaDevice) -> Result<(), CudaError> {
    let index: i8 = device.index().into();
    unsafe { cuda_check(hipSetDevice(index.into())) }
}
