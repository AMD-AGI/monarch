/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! RCCL bindings for ROCm - thin wrapper around RCCL C API
//!
//! Since RCCL is API-compatible with NCCL, we just call RCCL directly
//! via nccl_sys and provide Rust-friendly wrappers.

use std::mem::{MaybeUninit, transmute};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use torch_sys::{CudaDevice, ScalarType, TensorCell};

use crate::cuda_rocm::{CudaError, Stream, set_device};

// Re-export RCCL types from nccl_sys
pub use nccl_sys::{ncclComm_t, ncclUniqueId, ncclDataType_t, ncclRedOp_t};

// Helper to convert cuda_sys::hipStream_t to nccl_sys::hipStream_t
// They're the same underlying type but different Rust types
#[inline]
unsafe fn convert_stream(stream: cuda_sys::hipStream_t) -> nccl_sys::hipStream_t {
    transmute(stream)
}

/// RCCL configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NcclConfig {
    pub blocking: bool,
    pub cga_cluster_size: u8,
    pub min_ctas: u8,
    pub max_ctas: u8,
    pub net_name: Option<String>,
    pub split_share: bool,
}

impl Default for NcclConfig {
    fn default() -> Self {
        NcclConfig {
            blocking: true,
            cga_cluster_size: 4,
            min_ctas: 1,
            max_ctas: 32,
            net_name: None,
            split_share: false,
        }
    }
}

/// NCCL unique ID for communicator initialization
#[derive(Clone, Serialize, Deserialize)]
pub struct UniqueId {
    inner: ncclUniqueId,
}

impl std::fmt::Debug for UniqueId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UniqueId").finish()
    }
}

impl UniqueId {
    /// Create a new UniqueId using RCCL
    pub fn new() -> Result<Self, RawNcclError> {
        let mut inner = MaybeUninit::uninit();
        unsafe {
            nccl_check(nccl_sys::ncclGetUniqueId(inner.as_mut_ptr()))?;
            Ok(Self {
                inner: inner.assume_init(),
            })
        }
    }
}

/// NCCL reduction operations
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ReduceOp {
    Sum = 0,
    Prod = 1,
    Max = 2,
    Min = 3,
    Avg = 4,
}

impl From<ReduceOp> for ncclRedOp_t {
    fn from(op: ReduceOp) -> Self {
        Self(op as u32)
    }
}

/// NCCL data types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    Int8 = 0,
    Uint8 = 1,
    Int32 = 2,
    Uint32 = 3,
    Int64 = 4,
    Uint64 = 5,
    Float16 = 6,
    Float32 = 7,
    Float64 = 8,
    Bfloat16 = 9,
}

impl From<DataType> for ncclDataType_t {
    fn from(dt: DataType) -> Self {
        Self(dt as u32)
    }
}

impl TryFrom<ScalarType> for DataType {
    type Error = NcclError;

    fn try_from(value: ScalarType) -> Result<Self, Self::Error> {
        match value {
            ScalarType::Char => Ok(DataType::Int8),
            ScalarType::Byte => Ok(DataType::Uint8),
            ScalarType::Half => Ok(DataType::Float16),
            ScalarType::Float => Ok(DataType::Float32),
            ScalarType::Double => Ok(DataType::Float64),
            ScalarType::Int => Ok(DataType::Int32),
            ScalarType::Long => Ok(DataType::Int64),
            ScalarType::Bool => Ok(DataType::Uint8),
            ScalarType::BFloat16 => Ok(DataType::Bfloat16),
            ScalarType::Float8_e5m2 => Ok(DataType::Uint8),
            ScalarType::Float8_e4m3fn => Ok(DataType::Uint8),
            ScalarType::Float8_e4m3fnuz => Ok(DataType::Uint8),
            ScalarType::Float8_e5m2fnuz => Ok(DataType::Uint8),
            _ => Err(NcclError::InvalidDataType(value)),
        }
    }
}

/// NCCL error types
#[derive(Debug, Error)]
pub enum RawNcclError {
    #[error("a call to a CUDA function failed")]
    UnhandledCudaError,
    #[error("a call to the system failed")]
    SystemError,
    #[error("an internal check failed")]
    InternalError,
    #[error("an argument has an invalid value")]
    InvalidArgument,
    #[error("a call to NCCL is incorrect")]
    InvalidUsage,
    #[error("a call failed possibly due to a network error")]
    RemoteError,
}

/// NCCL error wrapper
#[derive(Debug, Error)]
pub enum NcclError {
    #[error("a NCCL-level error: {0:?}")]
    NcclError(#[from] RawNcclError),
    #[error("a CUDA-level error: {0:?}")]
    CudaError(#[from] CudaError),
    #[error("invalid NCCL data type: {0:#?}")]
    InvalidDataType(ScalarType),
    #[error("tensor used in collective must be contiguous")]
    NoncontiguousTensor,
    #[error("tensor must be on CUDA device")]
    InvalidDevice,
    #[error("got sparse tensor, only dense tensors allowed")]
    InvalidSparseTensor,
    #[error("undefined tensor used for NCCL operation")]
    UndefinedTensor,
}

/// NCCL status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NcclStatus {
    Success,
    InProgress,
}

fn nccl_check(result: nccl_sys::ncclResult_t) -> Result<NcclStatus, RawNcclError> {
    match result.0 {
        0 => Ok(NcclStatus::Success),
        1 => Err(RawNcclError::UnhandledCudaError),
        2 => Err(RawNcclError::SystemError),
        3 => Err(RawNcclError::InternalError),
        4 => Err(RawNcclError::InvalidArgument),
        5 => Err(RawNcclError::InvalidUsage),
        6 => Err(RawNcclError::RemoteError),
        7 => Ok(NcclStatus::InProgress),
        _ => panic!("Unknown ncclResult_t: {:?}", result.0),
    }
}

/// NCCL group ticket
pub struct NcclGroupTicket {
    _marker: std::marker::PhantomData<*const ()>,
}

/// Start a new NCCL group
pub fn group_start() -> Result<NcclGroupTicket, NcclError> {
    unsafe {
        nccl_check(nccl_sys::ncclGroupStart())?;
    }
    Ok(NcclGroupTicket {
        _marker: std::marker::PhantomData,
    })
}

/// End the NCCL group
pub fn group_end(_ticket: NcclGroupTicket) -> Result<(), NcclError> {
    unsafe {
        nccl_check(nccl_sys::ncclGroupEnd())?;
    }
    Ok(())
}

fn check_tensor(tensor: &torch_sys::Tensor) -> Result<(), NcclError> {
    if !tensor.defined() {
        return Err(NcclError::UndefinedTensor);
    }
    if !tensor.is_cuda() {
        return Err(NcclError::InvalidDevice);
    }
    if tensor.is_sparse() {
        return Err(NcclError::InvalidSparseTensor);
    }
    if !tensor.is_contiguous(torch_sys::suggest_memory_format(tensor)) {
        return Err(NcclError::NoncontiguousTensor);
    }
    Ok(())
}

/// RCCL Communicator - wraps RCCL comm_t
#[derive(Debug)]
pub struct Communicator {
    inner: ncclComm_t,
    world_size: i32,
    rank: i32,
    global_world_size: i32,
    global_rank: i32,
    device: CudaDevice,
}

unsafe impl Send for Communicator {}
unsafe impl Sync for Communicator {}

impl Communicator {
    /// Create a new communicator using RCCL
    pub fn new(
        device: CudaDevice,
        world_size: i32,
        unique_id: UniqueId,
        rank: i32,
    ) -> Result<Self, NcclError> {
        set_device(device)?;
        let mut inner = MaybeUninit::uninit();
        unsafe {
            nccl_check(nccl_sys::ncclCommInitRank(
                inner.as_mut_ptr(),
                world_size,
                unique_id.inner,
                rank,
            ))?;
            Ok(Self {
                inner: inner.assume_init(),
                world_size,
                rank,
                global_rank: rank,
                global_world_size: world_size,
                device,
            })
        }
    }

    /// All-reduce operation using RCCL
    pub fn all_reduce(
        &mut self,
        tensor: &TensorCell,
        reduce_op: ReduceOp,
        stream: &Stream,
    ) -> Result<NcclStatus, NcclError> {
        let tensor = tensor.borrow_mut();
        check_tensor(&tensor)?;
        let data_type: DataType = tensor.scalar_type().try_into()?;

        unsafe {
            Ok(nccl_check(nccl_sys::ncclAllReduce(
                tensor.data_ptr(),
                tensor.mut_data_ptr(),
                tensor.numel() as usize,
                data_type.into(),
                reduce_op.into(),
                self.inner,
                convert_stream(stream.stream()),
            ))?)
        }
    }

    /// Broadcast operation using RCCL
    pub fn broadcast(
        &mut self,
        tensor: &TensorCell,
        root: i32,
        stream: &Stream,
    ) -> Result<NcclStatus, NcclError> {
        let tensor = tensor.borrow_mut();
        check_tensor(&tensor)?;
        let data_type: DataType = tensor.scalar_type().try_into()?;

        unsafe {
            Ok(nccl_check(nccl_sys::ncclBroadcast(
                tensor.data_ptr(),
                tensor.mut_data_ptr(),
                tensor.numel() as usize,
                data_type.into(),
                root,
                self.inner,
                convert_stream(stream.stream()),
            ))?)
        }
    }

    /// Reduce operation using RCCL
    pub fn reduce(
        &mut self,
        tensor: &TensorCell,
        reduce_op: ReduceOp,
        root: i32,
        stream: &Stream,
    ) -> Result<NcclStatus, NcclError> {
        let tensor = tensor.borrow_mut();
        check_tensor(&tensor)?;
        let data_type: DataType = tensor.scalar_type().try_into()?;

        unsafe {
            Ok(nccl_check(nccl_sys::ncclReduce(
                tensor.data_ptr(),
                tensor.mut_data_ptr(),
                tensor.numel() as usize,
                data_type.into(),
                reduce_op.into(),
                root,
                self.inner,
                convert_stream(stream.stream()),
            ))?)
        }
    }

    /// Send operation using RCCL
    pub fn send(
        &mut self,
        tensor: &TensorCell,
        dst: i32,
        stream: &Stream,
    ) -> Result<NcclStatus, NcclError> {
        let tensor = tensor.borrow();
        check_tensor(&tensor)?;
        let data_type: DataType = tensor.scalar_type().try_into()?;

        unsafe {
            Ok(nccl_check(nccl_sys::ncclSend(
                tensor.data_ptr(),
                tensor.numel() as usize,
                data_type.into(),
                dst,
                self.inner,
                convert_stream(stream.stream()),
            ))?)
        }
    }

    /// Recv operation using RCCL
    pub fn recv(
        &mut self,
        tensor: &TensorCell,
        src: i32,
        stream: &Stream,
    ) -> Result<NcclStatus, NcclError> {
        let tensor = tensor.borrow_mut();
        check_tensor(&tensor)?;
        let data_type: DataType = tensor.scalar_type().try_into()?;

        unsafe {
            Ok(nccl_check(nccl_sys::ncclRecv(
                tensor.mut_data_ptr(),
                tensor.numel() as usize,
                data_type.into(),
                src,
                self.inner,
                convert_stream(stream.stream()),
            ))?)
        }
    }

    /// All-gather into tensor using RCCL
    pub fn all_gather_into_tensor(
        &mut self,
        output_cell: &TensorCell,
        input_cell: &TensorCell,
        stream: &Stream,
    ) -> Result<NcclStatus, NcclError> {
        let output = output_cell.borrow_mut();
        let input = if input_cell.aliases(output_cell) {
            unsafe { input_cell.get_unchecked() }
        } else {
            &input_cell.borrow()
        };

        check_tensor(&output)?;
        check_tensor(input)?;

        let data_type: DataType = input.scalar_type().try_into()?;

        unsafe {
            Ok(nccl_check(nccl_sys::ncclAllGather(
                input.data_ptr(),
                output.mut_data_ptr(),
                input.numel() as usize,
                data_type.into(),
                self.inner,
                convert_stream(stream.stream()),
            ))?)
        }
    }

    /// Reduce-scatter tensor using RCCL
    pub fn reduce_scatter_tensor(
        &mut self,
        output_cell: &TensorCell,
        input_cell: &TensorCell,
        reduce_op: ReduceOp,
        stream: &Stream,
    ) -> Result<NcclStatus, NcclError> {
        let output = output_cell.borrow_mut();
        let input = if input_cell.aliases(output_cell) {
            unsafe { input_cell.get_unchecked() }
        } else {
            &input_cell.borrow()
        };

        check_tensor(&output)?;
        check_tensor(input)?;

        let data_type: DataType = input.scalar_type().try_into()?;

        unsafe {
            Ok(nccl_check(nccl_sys::ncclReduceScatter(
                input.data_ptr(),
                output.mut_data_ptr(),
                output.numel() as usize,
                data_type.into(),
                reduce_op.into(),
                self.inner,
                convert_stream(stream.stream()),
            ))?)
        }
    }

    /// All-gather operation (list of tensors)
    pub fn all_gather(
        &mut self,
        output_cells: &[TensorCell],
        input_cell: &TensorCell,
        stream: &Stream,
    ) -> Result<NcclStatus, NcclError> {
        let output: Vec<_> = output_cells.iter().map(|t| t.borrow_mut()).collect();
        let input = input_cell.borrow();
        check_tensor(&input)?;

        let data_type: DataType = input.scalar_type().try_into()?;

        unsafe {
            nccl_check(nccl_sys::ncclGroupStart())?;
            for (i, out_tensor) in output.iter().enumerate() {
                let rank = i as i32;
                let output_ptr = out_tensor.mut_data_ptr();
                if rank == self.rank {
                    nccl_check(nccl_sys::ncclBroadcast(
                        input.data_ptr(),
                        output_ptr,
                        input.numel() as usize,
                        data_type.into(),
                        rank,
                        self.inner,
                        convert_stream(stream.stream()),
                    ))?;
                } else {
                    nccl_check(nccl_sys::ncclBroadcast(
                        output_ptr,
                        output_ptr,
                        out_tensor.numel() as usize,
                        data_type.into(),
                        rank,
                        self.inner,
                        convert_stream(stream.stream()),
                    ))?;
                }
            }
            nccl_check(nccl_sys::ncclGroupEnd())?;
        }
        Ok(NcclStatus::Success)
    }

    /// All-to-all single operation
    pub fn all_to_all_single(
        &mut self,
        output_cell: &TensorCell,
        input_cell: &TensorCell,
        stream: &Stream,
    ) -> Result<NcclStatus, NcclError> {
        let output = output_cell.borrow_mut();
        let input = if input_cell.aliases(output_cell) {
            unsafe { input_cell.get_unchecked() }
        } else {
            &input_cell.borrow()
        };

        check_tensor(&output)?;
        check_tensor(input)?;

        let data_type: DataType = input.scalar_type().try_into()?;
        let count = input.numel() as usize / self.world_size as usize;
        let rank_stride = input.nbytes() as isize / self.world_size as isize;

        unsafe {
            let send_buff = input.data_ptr();
            let recv_buff = output.mut_data_ptr();

            nccl_check(nccl_sys::ncclGroupStart())?;
            for r in 0..self.world_size {
                nccl_check(nccl_sys::ncclSend(
                    send_buff.offset(r as isize * rank_stride),
                    count,
                    data_type.into(),
                    r,
                    self.inner,
                    convert_stream(stream.stream()),
                ))?;
                nccl_check(nccl_sys::ncclRecv(
                    recv_buff.offset(r as isize * rank_stride),
                    count,
                    data_type.into(),
                    r,
                    self.inner,
                    convert_stream(stream.stream()),
                ))?;
            }
            nccl_check(nccl_sys::ncclGroupEnd())?;
        }
        Ok(NcclStatus::Success)
    }

    /// Barrier operation
    pub fn barrier(&mut self, stream: &Stream) -> Result<NcclStatus, NcclError> {
        let tensor = torch_sys::factory_float_tensor(&[1.0], self.device.into());
        let data_type: DataType = tensor.scalar_type().try_into()?;

        unsafe {
            Ok(nccl_check(nccl_sys::ncclAllReduce(
                tensor.data_ptr(),
                tensor.mut_data_ptr(),
                tensor.numel() as usize,
                data_type.into(),
                ReduceOp::Sum.into(),
                self.inner,
                convert_stream(stream.stream()),
            ))?)
        }
    }

    /// Split communicator (placeholder - needs RCCL split support)
    pub fn split_all(&mut self, _config: Option<NcclConfig>) -> Result<Self, NcclError> {
        Err(NcclError::NcclError(RawNcclError::InvalidUsage))
    }

    /// Split from ranks (placeholder - needs RCCL split support)
    pub fn split_from(
        &mut self,
        _ranks: Vec<i32>,
        _config: Option<NcclConfig>,
    ) -> Result<Option<Self>, NcclError> {
        Err(NcclError::NcclError(RawNcclError::InvalidUsage))
    }
}
