/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use cxx::type_id;
use cxx::ExternType;

// Conditionally compile CUDA-specific types
#[cfg(cuda)]
/// SAFETY: bindings
unsafe impl ExternType for CUstream_st {
    type Id = type_id!("CUstream_st");
    type Kind = cxx::kind::Opaque;
}

// Conditionally compile ROCm/HIP-specific types
#[cfg(rocm)]
/// SAFETY: bindings
unsafe impl ExternType for ihipStream_t { // <-- Implement for the OPAQUE STRUCT
    type Id = type_id!("hipStream_t");   // <-- Map to the C++ TYPEDEF
    type Kind = cxx::kind::Opaque;
}

// When building with cargo, this is actually the lib.rs file for a crate.
// Include the generated bindings.rs and suppress lints.
#[allow(non_camel_case_types)]
#[allow(non_upper_case_globals)]
#[allow(non_snake_case)]
mod inner {
    #[cfg(cargo)]
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

pub use inner::*;

#[cfg(test)]
mod tests {
    use std::mem::MaybeUninit;

    use super::*;

    #[test]
    fn sanity() {
        // SAFETY: testing bindings
        unsafe {
            // Conditionally compile test code
            #[cfg(cuda)]
            {
                let mut version = MaybeUninit::<i32>::uninit();
                let result = cuDriverGetVersion(version.as_mut_ptr());
                assert_eq!(result, cudaError_enum(0));
            }
            #[cfg(rocm)]
            {
                let mut version = MaybeUninit::<i32>::uninit(); // <-- Fix typo
                let result = hipDriverGetVersion(version.as_mut_ptr());
                assert_eq!(result, hipError_t(0)); // hipSuccess is 0
            }
        }
    }
}
