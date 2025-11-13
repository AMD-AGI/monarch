/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

fn main() {
    println!("cargo::rustc-check-cfg=cfg(rocm)");

    let use_rocm = build_utils::use_rocm();

    if use_rocm {
        println!("cargo:rustc-cfg=rocm");
        println!("cargo:warning=monarch_rdma_auto: Using ROCm backend.");

        let config = build_utils::discover_hip_config().unwrap_or_else(|_| {
            build_utils::print_rocm_error_help();
            std::process::exit(1);
        });

        if let Some(rocm_home) = config.rocm_home {
            println!(
                "cargo:rustc-link-search=native={}/lib",
                rocm_home.display()
            );
        }
        println!("cargo:rustc-link-lib=dylib=amdhip64");
    } else {
        println!("cargo:rustc-cfg=cuda");
        println!("cargo:warning=monarch_rdma_auto: Using CUDA backend.");

        let config = build_utils::discover_cuda_config().unwrap_or_else(|_| {
            build_utils::print_cuda_error_help();
            std::process::exit(1);
        });

        if let Some(cuda_home) = config.cuda_home {
            println!(
                "cargo:rustc-link-search=native={}/lib64",
                cuda_home.display()
            );
        }
        println!("cargo:rustc-link-lib=dylib=cuda");
    }
}
