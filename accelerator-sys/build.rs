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
    println!("cargo::rustc-check-cfg=cfg(cuda)");
    println!("cargo::rustc-check-cfg=cfg(rocm)");

    // --- DEBUG LINES ---
    println!("cargo:warning=-------------------------------------------------");
    let rocm_var = std::env::var("USE_ROCM").unwrap_or_else(|_| "NOT SET".to_string());
    println!("cargo:warning=DEBUG: USE_ROCM is '{}'", rocm_var);
    println!("cargo:warning=-------------------------------------------------");

    // This line is already called by build_utils, but is good for clarity
    println!("cargo:rerun-if-env-changed=USE_ROCM");

    let use_rocm = build_utils::use_rocm();

    // --- Bindgen Setup (Initialize first) ---
    let mut builder = bindgen::Builder::default()
        .header("src/wrapper.h") // This file must use #ifdef __HIP_PLATFORM_AMD__
        .clang_arg("-x")
        .clang_arg("c++")
        .clang_arg("-std=gnu++20")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .allowlist_function("(cu|CU|hip|HIP).*")
        .allowlist_type("(cu|CU|hip|HIP).*")
        .default_enum_style(bindgen::EnumVariation::NewType {
            is_bitfield: false,
            is_global: false,
        });

    let (mut lib_dirs, link_libs) = if use_rocm {
        // --- ROCm/HIP Configuration ---
        println!("cargo:rustc-cfg=rocm");

        let config = build_utils::discover_hip_config().unwrap_or_else(|_| {
            build_utils::print_rocm_error_help();
            std::process::exit(1);
        });

        let lib_dir = build_utils::get_rocm_lib_dir().unwrap_or_else(|_| {
            build_utils::print_rocm_lib_error_help();
            std::process::exit(1);
        });

        // --- FIX: Pass ALL discovered include paths to bindgen ---
        builder = builder.clang_arg("-D__HIP_PLATFORM_AMD__"); // Define for wrapper.h
        for include_dir in &config.include_dirs {
            builder = builder.clang_arg(format!("-I{}", include_dir.display()));
        }

        (vec![lib_dir], vec!["amdhip64"])
    } else {
        // --- CUDA Configuration ---
        println!("cargo:rustc-cfg=cuda");

        let config = build_utils::discover_cuda_config().unwrap_or_else(|_| {
            build_utils::print_cuda_error_help();
            std::process::exit(1);
        });

        let lib_dir = build_utils::get_cuda_lib_dir().unwrap_or_else(|_| {
            build_utils::print_cuda_lib_error_help();
            std::process::exit(1);
        });

        // --- FIX: Pass ALL discovered include paths to bindgen ---
        for include_dir in &config.include_dirs {
            builder = builder.clang_arg(format!("-I{}", include_dir.display()));
        }

        (vec![lib_dir], vec!["cuda", "cudart"])
    };

    // --- Python Include Setup ---
    let python_config = match build_utils::python_env_dirs_with_interpreter("python3") {
        Ok(config) => config,
        Err(_) => {
            eprintln!("Warning: Failed to get Python environment directories");
            build_utils::PythonConfig {
                include_dir: None,
                lib_dir: None,
            }
        }
    };

    if let Some(include_dir) = &python_config.include_dir {
        builder = builder.clang_arg(format!("-I{}", include_dir));
    }
    if let Some(lib_dir) = &python_config.lib_dir {
        lib_dirs.push(lib_dir.clone());
    }

    // --- Linking ---
    for lib_dir in lib_dirs {
        println!("cargo:rustc-link-search=native={}", lib_dir);
    }
    for lib in link_libs {
        println!("cargo:rustc-link-lib={}", lib);
    }

    // --- Generate Bindings ---
    let bindings = builder.generate().expect("Unable to generate bindings");
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings");

    println!("cargo:rustc-cfg=cargo");
    println!("cargo:rustc-check-cfg=cfg(cargo)");
}
