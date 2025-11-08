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
    // Try ROCm first, fall back to CUDA
    let (is_rocm, compute_lib_names) = if build_utils::find_rocm_home().is_some() {
        println!("cargo::warning=Using HIP from ROCm installation");
        (true, vec!["amdhip64"])
    } else {
        println!("cargo::warning=Using CUDA");
        (false, vec!["cuda", "cudart"])
    };

    // Discover compute configuration (CUDA or ROCm)
    let compute_config = if is_rocm {
        match build_utils::discover_rocm_config() {
            Ok(config) => build_utils::CudaConfig {
                cuda_home: config.rocm_home,
                include_dirs: config.include_dirs,
                lib_dirs: config.lib_dirs,
            },
            Err(_) => {
                build_utils::print_rocm_error_help();
                std::process::exit(1);
            }
        }
    } else {
        match build_utils::discover_cuda_config() {
            Ok(config) => config,
            Err(_) => {
                build_utils::print_cuda_error_help();
                std::process::exit(1);
            }
        }
    };

    // Start building the bindgen configuration
    let mut builder = bindgen::Builder::default()
        // The input header we would like to generate bindings for
        .header("src/wrapper.h")
        .clang_arg("-x")
        .clang_arg("c++")
        .clang_arg("-std=gnu++20")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        // Allow the specified functions and types
        .allowlist_function("cu.*")
        .allowlist_function("CU.*")
        .allowlist_type("cu.*")
        .allowlist_type("CU.*")
        // Use newtype enum style
        .default_enum_style(bindgen::EnumVariation::NewType {
            is_bitfield: false,
            is_global: false,
        });

    // Add CUDA/ROCm include paths from the discovered configuration
    for include_dir in &compute_config.include_dirs {
        builder = builder.clang_arg(format!("-I{}", include_dir.display()));
    }

    // Define HIP platform macros when using ROCm
    if is_rocm {
        builder = builder
            .clang_arg("-D__HIP_PLATFORM_AMD__=1")
            .clang_arg("-DUSE_ROCM=1")
            // Also allowlist HIP functions and types
            .allowlist_function("hip.*")
            .allowlist_type("hip.*");
    }

    // Include headers and libs from the active environment.
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
        println!("cargo::rustc-link-search=native={}", lib_dir);
        // Set cargo metadata to inform dependent binaries about how to set their
        // RPATH (see controller/build.rs for an example).
        println!("cargo::metadata=LIB_PATH={}", lib_dir);
    }

    // Get compute library directory and emit link directives
    let compute_lib_dir = if is_rocm {
        match build_utils::get_rocm_lib_dir() {
            Ok(dir) => dir,
            Err(_) => {
                build_utils::print_rocm_lib_error_help();
                std::process::exit(1);
            }
        }
    } else {
        match build_utils::get_cuda_lib_dir() {
            Ok(dir) => dir,
            Err(_) => {
                build_utils::print_cuda_lib_error_help();
                std::process::exit(1);
            }
        }
    };
    println!("cargo:rustc-link-search=native={}", compute_lib_dir);
    for lib_name in compute_lib_names {
        println!("cargo:rustc-link-lib={}", lib_name);
    }

    // Generate bindings - fail fast if this doesn't work
    let bindings = builder.generate().expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file
    match env::var("OUT_DIR") {
        Ok(out_dir) => {
            let out_path = PathBuf::from(out_dir);
            bindings
                .write_to_file(out_path.join("bindings.rs"))
                .expect("Couldn't write bindings");

            println!("cargo::rustc-cfg=cargo");
            println!("cargo::rustc-check-cfg=cfg(cargo)");
        }
        Err(_) => {
            println!("Note: OUT_DIR not set, skipping bindings file generation");
        }
    }
}
