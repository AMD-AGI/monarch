/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// Support both NCCL (CUDA) and RCCL (ROCm)
// RCCL provides an NCCL-compatible API
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__)
#include <rccl/rccl.h>
// Map CUDA stream type to HIP equivalent for RCCL
typedef struct ihipStream_t CUstream_st;
#else
#include <nccl.h>
#endif
