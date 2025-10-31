/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use cxx::ExternType;
use cxx::type_id;

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

/// SAFETY: bindings
/// Note: hipStream_t is typically an opaque pointer type (ihipStream_t*)
/// We implement ExternType for the underlying struct if needed by cxx
unsafe impl ExternType for ihipStream_t {
    type Id = type_id!("ihipStream_t");
    type Kind = cxx::kind::Opaque;
}

#[cfg(test)]
mod tests {
    use std::mem::MaybeUninit;

    use super::*;

    #[test]
    fn sanity() {
        // SAFETY: testing bindings
        unsafe {
            let mut version = MaybeUninit::<i32>::uninit();
            let result = hipDriverGetVersion(version.as_mut_ptr());
            assert_eq!(result, hipError_t::hipSuccess);
        }
    }
}
