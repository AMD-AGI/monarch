/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// Support both CUDA and HIP (ROCm)
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__)
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
// HIP provides CUDA-compatible types and functions
// Map CUDA stream type to HIP equivalent
typedef struct ihipStream_t CUstream_st;
#else
#include <cuda.h>
#include <cuda_runtime.h>
#endif
