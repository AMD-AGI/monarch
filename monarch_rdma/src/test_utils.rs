/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::sync::Once;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

/// Cached result of CUDA availability check
static CUDA_AVAILABLE: AtomicBool = AtomicBool::new(false);
static INIT: Once = Once::new();

/// Safely checks if CUDA is available on the system.
///
/// This function attempts to initialize CUDA and determine if it's available.
/// The result is cached after the first call, so subsequent calls are very fast.
///
/// # Returns
///
/// `true` if CUDA is available and can be initialized, `false` otherwise.
///
/// # Examples
///
/// ```
/// use monarch_rdma::is_cuda_available;
///
/// if is_cuda_available() {
///     println!("CUDA is available, can use GPU features");
/// } else {
///     println!("CUDA is not available, falling back to CPU-only mode");
/// }
/// ```
pub fn is_cuda_available() -> bool {
    INIT.call_once(|| {
        let available = check_cuda_available();
        CUDA_AVAILABLE.store(available, Ordering::SeqCst);
    });
    CUDA_AVAILABLE.load(Ordering::SeqCst)
}

/// Internal function that performs the actual CUDA/HIP availability check
fn check_cuda_available() -> bool {
    unsafe {
        // Try to initialize HIP (HIP auto-initializes, but we call for consistency)
        let result = cuda_sys::hipInit(0);

        if result != cuda_sys::hipError_t::hipSuccess {
            return false;
        }

        // Check if there are any HIP devices
        let mut device_count: i32 = 0;
        let count_result = cuda_sys::hipGetDeviceCount(&mut device_count);

        if count_result != cuda_sys::hipError_t::hipSuccess || device_count <= 0 {
            return false;
        }

        // Try to get the first device to verify it's actually accessible
        let mut device: i32 = 0;
        let device_result = cuda_sys::hipDeviceGet(&mut device, 0);

        if device_result != cuda_sys::hipError_t::hipSuccess {
            return false;
        }

        true
    }
}

#[cfg(test)]
pub mod test_utils {
    use std::time::Duration;
    use std::time::Instant;

    use hyperactor::ActorRef;
    use hyperactor::Instance;
    use hyperactor::Proc;
    use hyperactor::channel::ChannelTransport;
    use hyperactor::clock::Clock;
    use hyperactor::clock::RealClock;
    use hyperactor_mesh::Mesh;
    use hyperactor_mesh::ProcMesh;
    use hyperactor_mesh::RootActorMesh;
    use hyperactor_mesh::alloc::AllocSpec;
    use hyperactor_mesh::alloc::Allocator;
    use hyperactor_mesh::alloc::LocalAllocator;
    use ndslice::extent;

    use crate::IbverbsConfig;
    use crate::RdmaBuffer;
    use crate::cu_check;
    use crate::rdma_components::PollTarget;
    use crate::rdma_components::RdmaQueuePair;
    use crate::rdma_manager_actor::RdmaManagerActor;
    use crate::rdma_manager_actor::RdmaManagerMessageClient;
    use crate::validate_execution_context;
    // Waits for the completion of an RDMA operation.

    // This function polls for the completion of an RDMA operation by repeatedly
    // sending a `PollCompletion` message to the specified actor mesh and checking
    // the returned work completion status. It continues polling until the operation
    // completes or the specified timeout is reached.

    pub async fn wait_for_completion(
        qp: &mut RdmaQueuePair,
        poll_target: PollTarget,
        timeout_secs: u64,
    ) -> Result<bool, anyhow::Error> {
        let timeout = Duration::from_secs(timeout_secs);
        let start_time = Instant::now();
        while start_time.elapsed() < timeout {
            match qp.poll_completion_target(poll_target) {
                Ok(Some(_wc)) => {
                    return Ok(true);
                }
                Ok(None) => {
                    RealClock.sleep(Duration::from_millis(1)).await;
                }
                Err(e) => {
                    return Err(anyhow::anyhow!(e));
                }
            }
        }
        Err(anyhow::Error::msg("Timeout while waiting for completion"))
    }

    /// Posts a work request to the send queue of the given RDMA queue pair.
    pub async fn send_wqe_gpu(
        qp: &mut RdmaQueuePair,
        lhandle: &RdmaBuffer,
        rhandle: &RdmaBuffer,
        op_type: u32,
    ) -> Result<(), anyhow::Error> {
        unsafe {
            let ibv_qp = qp.qp as *mut rdmaxcel_sys::ibv_qp;
            let dv_qp = qp.dv_qp as *mut rdmaxcel_sys::mlx5dv_qp;
            let params = rdmaxcel_sys::wqe_params_t {
                laddr: lhandle.addr,
                length: lhandle.size,
                lkey: lhandle.lkey,
                wr_id: qp.send_wqe_idx,
                signaled: true,
                op_type,
                raddr: rhandle.addr,
                rkey: rhandle.rkey,
                qp_num: (*ibv_qp).qp_num,
                buf: (*dv_qp).sq.buf as *mut u8,
                wqe_cnt: (*dv_qp).sq.wqe_cnt,
                dbrec: (*dv_qp).dbrec,
                ..Default::default()
            };
            rdmaxcel_sys::launch_send_wqe(params);
            qp.send_wqe_idx += 1;
        }
        Ok(())
    }

    /// Posts a work request to the receive queue of the given RDMA queue pair.
    pub async fn recv_wqe_gpu(
        qp: &mut RdmaQueuePair,
        lhandle: &RdmaBuffer,
        _rhandle: &RdmaBuffer,
        op_type: u32,
    ) -> Result<(), anyhow::Error> {
        // Populate params using lhandle and rhandle
        unsafe {
            let ibv_qp = qp.qp as *mut rdmaxcel_sys::ibv_qp;
            let dv_qp = qp.dv_qp as *mut rdmaxcel_sys::mlx5dv_qp;
            let params = rdmaxcel_sys::wqe_params_t {
                laddr: lhandle.addr,
                length: lhandle.size,
                lkey: lhandle.lkey,
                wr_id: qp.recv_wqe_idx,
                op_type,
                signaled: true,
                qp_num: (*ibv_qp).qp_num,
                buf: (*dv_qp).rq.buf as *mut u8,
                wqe_cnt: (*dv_qp).rq.wqe_cnt,
                dbrec: (*dv_qp).dbrec,
                ..Default::default()
            };
            rdmaxcel_sys::launch_recv_wqe(params);
            qp.recv_wqe_idx += 1;
            qp.recv_db_idx += 1;
        }
        Ok(())
    }

    pub async fn ring_db_gpu(qp: &mut RdmaQueuePair) -> Result<(), anyhow::Error> {
        RealClock.sleep(Duration::from_millis(2)).await;
        unsafe {
            let dv_qp = qp.dv_qp as *mut rdmaxcel_sys::mlx5dv_qp;
            let base_ptr = (*dv_qp).sq.buf as *mut u8;
            let wqe_cnt = (*dv_qp).sq.wqe_cnt;
            let stride = (*dv_qp).sq.stride;
            if (wqe_cnt as u64) < (qp.send_wqe_idx - qp.send_db_idx) {
                return Err(anyhow::anyhow!("Overflow of WQE, possible data loss"));
            }
            while qp.send_db_idx < qp.send_wqe_idx {
                let offset = (qp.send_db_idx % wqe_cnt as u64) * stride as u64;
                let src_ptr = (base_ptr as *mut u8).wrapping_add(offset as usize);
                rdmaxcel_sys::launch_db_ring((*dv_qp).bf.reg, src_ptr as *mut std::ffi::c_void);
                qp.send_db_idx += 1;
            }
        }
        Ok(())
    }

    /// Wait for completion on a specific completion queue
    pub async fn wait_for_completion_gpu(
        qp: &mut RdmaQueuePair,
        poll_target: PollTarget,
        timeout_secs: u64,
    ) -> Result<bool, anyhow::Error> {
        let timeout = Duration::from_secs(timeout_secs);
        let start_time = Instant::now();

        while start_time.elapsed() < timeout {
            // Get the appropriate completion queue and index based on the poll target
            let (cq, idx, cq_type_str) = match poll_target {
                PollTarget::Send => (
                    qp.dv_send_cq as *mut rdmaxcel_sys::mlx5dv_cq,
                    qp.send_cq_idx,
                    "send",
                ),
                PollTarget::Recv => (
                    qp.dv_recv_cq as *mut rdmaxcel_sys::mlx5dv_cq,
                    qp.recv_cq_idx,
                    "receive",
                ),
            };

            // Poll the completion queue
            let result =
                unsafe { rdmaxcel_sys::launch_cqe_poll(cq as *mut std::ffi::c_void, idx as i32) };

            match result {
                rdmaxcel_sys::CQE_POLL_TRUE => {
                    // Update the appropriate index based on the poll target
                    match poll_target {
                        PollTarget::Send => qp.send_cq_idx += 1,
                        PollTarget::Recv => qp.recv_cq_idx += 1,
                    }
                    return Ok(true);
                }
                rdmaxcel_sys::CQE_POLL_ERROR => {
                    return Err(anyhow::anyhow!("Error polling {} completion", cq_type_str));
                }
                _ => {
                    // No completion yet, sleep and try again
                    RealClock.sleep(Duration::from_millis(1)).await;
                }
            }
        }

        Err(anyhow::Error::msg("Timeout while waiting for completion"))
    }

    pub struct RdmaManagerTestEnv<'a> {
        buffer_1: Buffer,
        buffer_2: Buffer,
        pub client_1: &'a Instance<()>,
        pub client_2: &'a Instance<()>,
        pub actor_1: ActorRef<RdmaManagerActor>,
        pub actor_2: ActorRef<RdmaManagerActor>,
        pub rdma_handle_1: RdmaBuffer,
        pub rdma_handle_2: RdmaBuffer,
        cuda_context_1: Option<cuda_sys::hipCtx_t>,
        cuda_context_2: Option<cuda_sys::hipCtx_t>,
    }

    #[derive(Debug, Clone)]
    pub struct Buffer {
        ptr: u64,
        len: usize,
        #[allow(dead_code)]
        cpu_ref: Option<Box<[u8]>>,
        // Track if this is a hipMalloc allocation (true) or hipMemCreate/VMM (false)
        is_hip_malloc: bool,
    }
    /// Helper function to parse accelerator strings
    async fn parse_accel(accel: &str, config: &mut IbverbsConfig) -> (String, usize) {
        let (backend, idx) = accel.split_once(':').unwrap();
        let parsed_idx = idx.parse::<usize>().unwrap();

        if backend == "cuda" {
            config.use_gpu_direct = validate_execution_context().await.is_ok();
        }

        (backend.to_string(), parsed_idx)
    }

    impl RdmaManagerTestEnv<'_> {
        /// Sets up the RDMA test environment with a specified QP type.
        ///
        /// This function initializes the RDMA test environment by setting up two actor meshes
        /// with their respective RDMA configurations. It also prepares two buffers for testing
        /// RDMA operations and fills the first buffer with test data.
        ///
        /// # Arguments
        ///
        /// * `buffer_size` - The size of the buffers to be used in the test.
        /// * `accel1` - Accelerator for first actor (e.g., "cpu:0", "cuda:0")
        /// * `accel2` - Accelerator for second actor (e.g., "cpu:0", "cuda:1")
        /// * `qp_type` - The queue pair type to use (Auto, Standard, or Mlx5dv)
        pub async fn setup_with_qp_type(
            buffer_size: usize,
            accel1: &str,
            accel2: &str,
            qp_type: crate::ibverbs_primitives::RdmaQpType,
        ) -> Result<Self, anyhow::Error> {
            eprintln!("[DEBUG] setup_with_qp_type: START accel1={}, accel2={}, qp_type={:?}", accel1, accel2, qp_type);
            // Use device selection logic to find optimal RDMA devices
            let mut config1 = IbverbsConfig::targeting(accel1);
            let mut config2 = IbverbsConfig::targeting(accel2);

            // Set the QP type
            config1.qp_type = qp_type;
            config2.qp_type = qp_type;

            eprintln!("[DEBUG] setup_with_qp_type: About to parse accels");
            let parsed_accel1 = parse_accel(accel1, &mut config1).await;
            let parsed_accel2 = parse_accel(accel2, &mut config2).await;
            eprintln!("[DEBUG] setup_with_qp_type: Parsed accels - accel1={:?}, accel2={:?}", parsed_accel1, parsed_accel2);

            let alloc_1 = LocalAllocator
                .allocate(AllocSpec {
                    extent: extent! { proc = 1 },
                    constraints: Default::default(),
                    proc_name: None,
                    transport: ChannelTransport::Local,
                })
                .await
                .unwrap();

            eprintln!("[DEBUG] setup_with_qp_type: About to create proc instance");
            let (instance, _) = Proc::local().instance("test").unwrap();

            eprintln!("[DEBUG] setup_with_qp_type: About to allocate proc_mesh_1");
            let proc_mesh_1 = Box::leak(Box::new(ProcMesh::allocate(alloc_1).await.unwrap()));
            eprintln!("[DEBUG] setup_with_qp_type: About to spawn actor_mesh_1 with config1");
            let actor_mesh_1: RootActorMesh<'_, RdmaManagerActor> = proc_mesh_1
                .spawn(&instance, "rdma_manager", &Some(config1))
                .await
                .unwrap();
            eprintln!("[DEBUG] setup_with_qp_type: actor_mesh_1 spawned successfully");

            let alloc_2 = LocalAllocator
                .allocate(AllocSpec {
                    extent: extent! { proc = 1 },
                    constraints: Default::default(),
                    proc_name: None,
                    transport: ChannelTransport::Local,
                })
                .await
                .unwrap();

            eprintln!("[DEBUG] setup_with_qp_type: About to allocate proc_mesh_2");
            let proc_mesh_2 = Box::leak(Box::new(ProcMesh::allocate(alloc_2).await.unwrap()));
            eprintln!("[DEBUG] setup_with_qp_type: About to spawn actor_mesh_2 with config2");
            let actor_mesh_2: RootActorMesh<'_, RdmaManagerActor> = proc_mesh_2
                .spawn(&instance, "rdma_manager", &Some(config2))
                .await
                .unwrap();
            eprintln!("[DEBUG] setup_with_qp_type: actor_mesh_2 spawned successfully");

            eprintln!("[DEBUG] setup_with_qp_type: About to allocate buffers");
            let mut buf_vec = Vec::new();
            let mut cuda_contexts = Vec::new();

            for (idx, accel) in [parsed_accel1.clone(), parsed_accel2.clone()].iter().enumerate() {
                eprintln!("[DEBUG] setup_with_qp_type: Processing buffer {} for accel {:?}", idx, accel);
                if accel.0 == "cpu" {
                    let mut buffer = vec![0u8; buffer_size].into_boxed_slice();
                    buf_vec.push(Buffer {
                        ptr: buffer.as_mut_ptr() as u64,
                        len: buffer.len(),
                        cpu_ref: Some(buffer),
                        is_hip_malloc: false,
                    });
                    cuda_contexts.push(None);
                    continue;
                }
                // HIP/ROCm case
                unsafe {
                    eprintln!("[DEBUG] setup_with_qp_type: HIP buffer allocation starting");
                    cu_check!(cuda_sys::hipInit(0));
                    eprintln!("[DEBUG] setup_with_qp_type: hipInit done");

                    let mut dptr: *mut std::ffi::c_void = std::ptr::null_mut();
                    let mut handle: cuda_sys::hipMemGenericAllocationHandle_t = std::mem::zeroed();

                    let mut device: i32 = accel.1 as i32;
                    cu_check!(cuda_sys::hipDeviceGet(&mut device, accel.1 as i32));
                    eprintln!("[DEBUG] setup_with_qp_type: hipDeviceGet done, device={}", device);

                    let mut context: cuda_sys::hipCtx_t = std::ptr::null_mut();
                    cu_check!(cuda_sys::hipCtxCreate(&mut context, 0, device));
                    eprintln!("[DEBUG] setup_with_qp_type: hipCtxCreate done");
                    cu_check!(cuda_sys::hipCtxSetCurrent(context));
                    eprintln!("[DEBUG] setup_with_qp_type: hipCtxSetCurrent done");

                    // For Standard QP on ROCm < 7.0, use hipMalloc instead of hipMemCreate
                    // because HSA dmabuf export only works with hipMalloc allocations on ROCm 6.x
                    // ROCm 7.0+ has hipMemGetHandleForAddressRange which works with hipMemCreate
                    let use_hip_malloc = matches!(qp_type, crate::ibverbs_primitives::RdmaQpType::Standard);

                    if use_hip_malloc {
                        eprintln!("[DEBUG] setup_with_qp_type: Using hipMalloc for Standard QP on ROCm < 7.0");
                        cu_check!(cuda_sys::hipMalloc(&mut dptr, buffer_size));
                        eprintln!("[DEBUG] setup_with_qp_type: hipMalloc done, ptr={:p}", dptr);

                        buf_vec.push(Buffer {
                            ptr: dptr as u64,
                            len: buffer_size,
                            cpu_ref: None,
                            is_hip_malloc: true,
                        });
                        eprintln!("[DEBUG] setup_with_qp_type: Buffer {} pushed", idx);
                        cuda_contexts.push(Some(context));
                        continue;
                    }

                    eprintln!("[DEBUG] setup_with_qp_type: Using hipMemCreate/hipMemMap allocation");
                    eprintln!("[DEBUG] setup_with_qp_type: About to setup hipMemAllocationProp");
                    let mut granularity: usize = 0;
                    let mut prop: cuda_sys::hipMemAllocationProp = std::mem::zeroed();
                    prop.type_ = cuda_sys::hipMemAllocationType::hipMemAllocationTypePinned;
                    prop.location.type_ = cuda_sys::hipMemLocationType::hipMemLocationTypeDevice;
                    prop.location.id = device;
                    prop.allocFlags.gpuDirectRDMACapable = 1;
                    prop.requestedHandleType =
                        cuda_sys::hipMemAllocationHandleType::hipMemHandleTypePosixFileDescriptor;

                    eprintln!("[DEBUG] setup_with_qp_type: About to call hipMemGetAllocationGranularity");
                    cu_check!(cuda_sys::hipMemGetAllocationGranularity(
                        &mut granularity as *mut usize,
                        &prop,
                        cuda_sys::hipMemAllocationGranularity_flags::hipMemAllocationGranularityMinimum,
                    ));
                    eprintln!("[DEBUG] setup_with_qp_type: hipMemGetAllocationGranularity done, granularity={}", granularity);

                    // ensure our size is aligned
                    let /*mut*/ padded_size: usize = ((buffer_size - 1) / granularity + 1) * granularity;
                    assert!(padded_size == buffer_size);

                    eprintln!("[DEBUG] setup_with_qp_type: About to call hipMemCreate, size={}", padded_size);
                    cu_check!(cuda_sys::hipMemCreate(
                        &mut handle as *mut cuda_sys::hipMemGenericAllocationHandle_t,
                        padded_size,
                        &prop,
                        0
                    ));
                    eprintln!("[DEBUG] setup_with_qp_type: hipMemCreate done");
                    // reserve and map the memory
                    eprintln!("[DEBUG] setup_with_qp_type: About to call hipMemAddressReserve");
                    cu_check!(cuda_sys::hipMemAddressReserve(
                        &mut dptr,
                        padded_size,
                        0,
                        std::ptr::null_mut(),
                        0,
                    ));
                    eprintln!("[DEBUG] setup_with_qp_type: hipMemAddressReserve done");
                    assert!((dptr as usize).is_multiple_of(granularity));
                    assert!(padded_size.is_multiple_of(granularity));

                    eprintln!("[DEBUG] setup_with_qp_type: About to call hipMemMap");
                    // fails if a add cu_check macro; but passes if we don't
                    let err = cuda_sys::hipMemMap(
                        dptr,
                        padded_size,
                        0,
                        handle as cuda_sys::hipMemGenericAllocationHandle_t,
                        0,
                    );
                    eprintln!("[DEBUG] setup_with_qp_type: hipMemMap returned {:?}", err);
                    if err != cuda_sys::hipError_t::hipSuccess {
                        panic!("failed reserving and mapping memory {:?}", err);
                    }
                    eprintln!("[DEBUG] setup_with_qp_type: hipMemMap completed successfully");

                    // set access
                    eprintln!("[DEBUG] setup_with_qp_type: About to setup access");
                    let mut access_desc: cuda_sys::hipMemAccessDesc = std::mem::zeroed();
                    access_desc.location.type_ =
                        cuda_sys::hipMemLocationType::hipMemLocationTypeDevice;
                    access_desc.location.id = device;
                    access_desc.flags =
                        cuda_sys::hipMemAccessFlags::hipMemAccessFlagsProtReadWrite;
                    eprintln!("[DEBUG] setup_with_qp_type: About to call hipMemSetAccess");
                    cu_check!(cuda_sys::hipMemSetAccess(dptr, padded_size, &access_desc, 1));
                    eprintln!("[DEBUG] setup_with_qp_type: hipMemSetAccess done");
                    buf_vec.push(Buffer {
                        ptr: dptr as u64,
                        len: padded_size,
                        cpu_ref: None,
                        is_hip_malloc: false,
                    });
                    eprintln!("[DEBUG] setup_with_qp_type: Buffer {} pushed", idx);
                    cuda_contexts.push(Some(context));
                }
            }
            eprintln!("[DEBUG] setup_with_qp_type: All buffers allocated");

            // Fill buffer1 with test data
            eprintln!("[DEBUG] setup_with_qp_type: About to fill buffer with test data");
            if parsed_accel1.0 == "cuda" {
                let mut temp_buffer = vec![0u8; buffer_size].into_boxed_slice();
                for (i, val) in temp_buffer.iter_mut().enumerate() {
                    *val = (i % 256) as u8;
                }
                unsafe {
                    eprintln!("[DEBUG] setup_with_qp_type: About to set HIP context for memcpy");
                    // Use the HIP context that was created for the first buffer
                    cu_check!(cuda_sys::hipCtxSetCurrent(
                        cuda_contexts[0].expect("No HIP context found")
                    ));

                    eprintln!("[DEBUG] setup_with_qp_type: About to hipMemcpyHtoD");
                    cu_check!(cuda_sys::hipMemcpyHtoD(
                        buf_vec[0].ptr as *mut std::ffi::c_void,
                        temp_buffer.as_ptr() as *mut std::ffi::c_void,
                        temp_buffer.len()
                    ));
                    eprintln!("[DEBUG] setup_with_qp_type: hipMemcpyHtoD done");
                }
            } else {
                unsafe {
                    let ptr = buf_vec[0].ptr as *mut u8; // or *const u8
                    for i in 0..buf_vec[0].len {
                        *ptr.add(i) = (i % 256) as u8;
                    }
                }
            }
            eprintln!("[DEBUG] setup_with_qp_type: About to get actors");
            let actor_1 = actor_mesh_1.get(0).unwrap();
            let actor_2 = actor_mesh_2.get(0).unwrap();

            eprintln!("[DEBUG] setup_with_qp_type: About to request_buffer from actor_1");
            let rdma_handle_1 = actor_1
                .request_buffer(proc_mesh_1.client(), buf_vec[0].ptr as usize, buffer_size)
                .await?;
            eprintln!("[DEBUG] setup_with_qp_type: rdma_handle_1 obtained");
            let rdma_handle_2 = actor_2
                .request_buffer(proc_mesh_2.client(), buf_vec[1].ptr as usize, buffer_size)
                .await?;
            // Get keys from both actors.

            let buffer_2 = buf_vec.remove(1);
            let buffer_1 = buf_vec.remove(0);
            Ok(Self {
                buffer_1,
                buffer_2,
                client_1: proc_mesh_1.client(),
                client_2: proc_mesh_2.client(),
                actor_1,
                actor_2,
                rdma_handle_1,
                rdma_handle_2,
                cuda_context_1: cuda_contexts.first().cloned().flatten(),
                cuda_context_2: cuda_contexts.get(1).cloned().flatten(),
            })
        }

        pub async fn cleanup(self) -> Result<(), anyhow::Error> {
            self.actor_1
                .release_buffer(self.client_1, self.rdma_handle_1.clone())
                .await?;
            self.actor_2
                .release_buffer(self.client_2, self.rdma_handle_2.clone())
                .await?;
            if self.cuda_context_1.is_some() {
                unsafe {
                    cu_check!(cuda_sys::hipCtxSetCurrent(
                        self.cuda_context_1.expect("No HIP context found")
                    ));
                    if self.buffer_1.is_hip_malloc {
                        // hipMalloc allocation - use hipFree
                        cu_check!(cuda_sys::hipFree(self.buffer_1.ptr as *mut std::ffi::c_void));
                    } else {
                        // VMM allocation - use hipMemUnmap + hipMemAddressFree
                        cu_check!(cuda_sys::hipMemUnmap(
                            self.buffer_1.ptr as cuda_sys::hipDeviceptr_t,
                            self.buffer_1.len
                        ));
                        cu_check!(cuda_sys::hipMemAddressFree(
                            self.buffer_1.ptr as cuda_sys::hipDeviceptr_t,
                            self.buffer_1.len
                        ));
                    }
                }
            }
            if self.cuda_context_2.is_some() {
                unsafe {
                    cu_check!(cuda_sys::hipCtxSetCurrent(
                        self.cuda_context_2.expect("No HIP context found")
                    ));
                    if self.buffer_2.is_hip_malloc {
                        // hipMalloc allocation - use hipFree
                        cu_check!(cuda_sys::hipFree(self.buffer_2.ptr as *mut std::ffi::c_void));
                    } else {
                        // VMM allocation - use hipMemUnmap + hipMemAddressFree
                        cu_check!(cuda_sys::hipMemUnmap(
                            self.buffer_2.ptr as cuda_sys::hipDeviceptr_t,
                            self.buffer_2.len
                        ));
                        cu_check!(cuda_sys::hipMemAddressFree(
                            self.buffer_2.ptr as cuda_sys::hipDeviceptr_t,
                            self.buffer_2.len
                        ));
                    }
                }
            }
            Ok(())
        }

        /// Sets up the RDMA test environment with auto-detected QP type.
        ///
        /// This is a convenience wrapper around `setup_with_qp_type` that uses
        /// `RdmaQpType::Auto` to automatically select the appropriate QP type.
        ///
        /// # Arguments
        ///
        /// * `buffer_size` - The size of the buffers to be used in the test.
        /// * `accel1` - Accelerator for first actor (e.g., "cpu:0", "cuda:0")
        /// * `accel2` - Accelerator for second actor (e.g., "cpu:0", "cuda:1")
        pub async fn setup(
            buffer_size: usize,
            accel1: &str,
            accel2: &str,
        ) -> Result<Self, anyhow::Error> {
            Self::setup_with_qp_type(
                buffer_size,
                accel1,
                accel2,
                crate::ibverbs_primitives::RdmaQpType::Auto,
            )
            .await
        }

        pub async fn verify_buffers(&self, size: usize) -> Result<(), anyhow::Error> {
            let mut buf_vec = Vec::new();
            for (virtual_addr, cuda_context) in [
                (self.buffer_1.ptr, self.cuda_context_1),
                (self.buffer_2.ptr, self.cuda_context_2),
            ] {
                if cuda_context.is_some() {
                    let mut temp_buffer = vec![0u8; size].into_boxed_slice();
                    // SAFETY: The buffer is allocated with the correct size and the pointer is valid.
                    unsafe {
                        cu_check!(cuda_sys::hipCtxSetCurrent(
                            cuda_context.expect("No HIP context found")
                        ));
                        cu_check!(cuda_sys::hipMemcpyDtoH(
                            temp_buffer.as_mut_ptr() as *mut std::ffi::c_void,
                            virtual_addr as cuda_sys::hipDeviceptr_t,
                            size
                        ));
                    }
                    buf_vec.push(Buffer {
                        ptr: temp_buffer.as_mut_ptr() as u64,
                        len: size,
                        cpu_ref: Some(temp_buffer),
                        is_hip_malloc: false,
                    });
                } else {
                    buf_vec.push(Buffer {
                        ptr: virtual_addr,
                        len: size,
                        cpu_ref: None,
                        is_hip_malloc: false,  // These are for setup(), not the main Standard QP tests
                    });
                }
            }
            // SAFETY: The pointers are valid and the buffers have the same length.
            unsafe {
                let ptr1 = buf_vec[0].ptr as *mut u8;
                let ptr2: *mut u8 = buf_vec[1].ptr as *mut u8;
                for i in 0..buf_vec[0].len {
                    if *ptr1.add(i) != *ptr2.add(i) {
                        return Err(anyhow::anyhow!("Buffers are not equal at index {}", i));
                    }
                }
            }
            Ok(())
        }
    }
}
