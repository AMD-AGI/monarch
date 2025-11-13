/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::env;
use std::path::PathBuf;

#[cfg(target_os = "macos")]
fn main() {}

#[cfg(not(target_os = "macos"))]
fn main() {
    println!("cargo::rustc-check-cfg=cfg(rocm)");
    println!("cargo::rustc-check-cfg=cfg(cuda)");

    // Determine platform from environment, not from a potentially unset cfg flag
    let use_rocm = build_utils::use_rocm();

    // Now, set the cfg flags for this crate based on the environment
    if use_rocm {
        println!("cargo:rustc-cfg=rocm");
    } else {
        println!("cargo:rustc-cfg=cuda");
    }

    // --- 1. Link Accelerator Libraries (from accelerator-sys) ---
    // These are the core driver libs.
    if use_rocm {
        println!("cargo:rustc-link-lib=amdhip64");
    } else {
        println!("cargo:rustc-link-lib=cuda");
        println!("cargo:rustc-link-lib=cudart");
    }

    // --- 2. Link RDMA Hardware Libraries (Always) ---
    println!("cargo:rustc-link-lib=ibverbs");
    println!("cargo:rustc-link-lib=mlx5");

    // --- 3. Link PyTorch & C10 Libraries ---
    let use_pytorch_apis = build_utils::get_env_var_with_rerun("TORCH_SYS_USE_PYTORCH_APIS")
        .unwrap_or_else(|_| "1".to_owned());

    if use_pytorch_apis == "1" {
        // Find the main PyTorch library path
        let python_interpreter = PathBuf::from("python3"); // Use python3
        if let Ok(output) = std::process::Command::new(&python_interpreter)
            .arg("-c")
            .arg(build_utils::PYTHON_PRINT_PYTORCH_DETAILS) // Assumes this script prints LIBTORCH_LIB
            .output()
        {
            if output.status.success() {
                for line in String::from_utf8_lossy(&output.stdout).lines() {
                    if let Some(path) = line.strip_prefix("LIBTORCH_LIB: ") {
                        // Add library search path and rpath
                        println!("cargo:rustc-link-search=native={}", path);
                        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", path);
                    }
                }
            } else {
                // Panic if python script fails
                panic!(
                    "Failed to get PyTorch details from python. Is torch installed in your python3 environment?\nStderr: {}",
                    String::from_utf8_lossy(&output.stderr)
                );
            }
        } else {
            panic!("Failed to run python3. Is it in your PATH?");
        }

        // Link common PyTorch libs
        println!("cargo:rustc-link-lib=torch_cpu");
        println!("cargo:rustc-link-lib=torch");
        println!("cargo:rustc-link-lib=c10");

        // Link platform-specific PyTorch libs (from torch-sys-accel)
        if use_rocm {
            println!("cargo:rustc-link-lib=c10_hip");
            // Add torch_hip if it exists
            // println!("cargo:rustc-link-lib=torch_hip");
        } else {
            println!("cargo:rustc-link-lib=c10_cuda");
            println!("cargo:rustc-link-lib=torch_cuda");
        }
    }

    // --- 4. Link our compiled 'rdmaxcel-sys' static libraries ---
    // We now have only one dependency crate: rdmaxcel-sys.
    // We get its OUT_DIR and conditionally link the lib name.

    // This is the variable Cargo sets from our dependency
    let dep_out_dir_var = "DEP_RDMAXCEL_SYS_OUT_DIR";

    if let Ok(rdmaxcel_out_dir) = env::var(dep_out_dir_var) {
        println!("cargo:rustc-link-search=native={}", rdmaxcel_out_dir);

        // Link the common C and C++ static libraries
        println!("cargo:rustc-link-lib=static=rdmaxcel");
        println!("cargo:rustc-link-lib=static=rdmaxcel_cpp");

        // Link the platform-specific accelerator static library
        if use_rocm {
            println!("cargo:rustc-link-lib=static=rdmaxcel_hip");
        } else {
            println!("cargo:rustc-link-lib=static=rdmaxcel_cuda");
        }
    } else {
        // This fallback logic is now simplified, as it only looks for one crate
        eprintln!(
            "Warning: {} not found. Using fallback path.",
            dep_out_dir_var
        );

        let (build_subdir, lib_name) = if use_rocm {
            ("hip_build", "rdmaxcel_hip")
        } else {
            ("cuda_build", "rdmaxcel_cuda")
        };

        // Link accelerator lib
        let accelerator_build_dir = format!("../rdmaxcel-sys/target/{}", build_subdir);
        println!("cargo:rustc-link-search=native={}", accelerator_build_dir);
        println!("cargo:rustc-link-lib=static={}", lib_name);

        // Find the common C/C++ libs from the most recent build
        let monarch_target_dir = "../target/debug/build";
        if let Ok(entries) = std::fs::read_dir(monarch_target_dir) {
            let search_prefix = "rdmaxcel-sys-"; // Only one prefix now

            let mut rdmaxcel_dirs: Vec<_> = entries
                .filter_map(|entry| entry.ok())
                .filter(|entry| {
                    let name = entry.file_name().to_string_lossy().to_string();
                    name.starts_with(search_prefix)
                })
                .collect();

            // Sort by modification time and use the most recent
            rdmaxcel_dirs
                .sort_by_key(|entry| entry.metadata().ok().and_then(|m| m.modified().ok()));

            if let Some(most_recent) = rdmaxcel_dirs.last() {
                let out_dir = most_recent.path().join("out");
                if out_dir.exists() {
                    println!("cargo:rustc-link-search=native={}", out_dir.display());
                    println!("cargo:rustc-link-lib=static=rdmaxcel");
                    println!("cargo:rustc-link-lib=static=rdmaxcel_cpp");
                }
            } else {
                eprintln!(
                    "Warning: No {} build directories found in fallback",
                    search_prefix
                );
            }
        }
    }

    // --- 5. Final Linker Arguments ---

    // Add rpath for Python libs if found
    if let Ok(py_config) = build_utils::python_env_dirs_with_interpreter("python3") {
        if let Some(lib_dir) = &py_config.lib_dir {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir);
        }
    }

    // Add rpath for NCCL libraries if available
    if let Ok(nccl_lib_path) = env::var("DEP_NCCL_LIB_PATH") {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", nccl_lib_path);
    }

    // Disable new dtags, as conda envs generally use `RPATH` over `RUNPATH`
    println!("cargo:rustc-link-arg=-Wl,--disable-new-dtags");

    // Set build configuration flags
    println!("cargo:rustc-cfg=cargo");
    println!("cargo:rustc-check-cfg=cfg(cargo)");
}
