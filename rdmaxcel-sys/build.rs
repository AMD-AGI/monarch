/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::env;
use std::path::Path;
use std::path::PathBuf;

#[cfg(target_os = "macos")]
fn main() {}

#[cfg(not(target_os = "macos"))]
fn main() {
    // Link against the ibverbs library
    println!("cargo:rustc-link-lib=ibverbs");

    // Link against the mlx5 library
    println!("cargo:rustc-link-lib=mlx5");

    // Tell cargo to invalidate the built crate whenever the wrapper changes
    println!("cargo:rerun-if-changed=src/rdmaxcel.h");
    println!("cargo:rerun-if-changed=src/rdmaxcel.c");
    println!("cargo:rerun-if-changed=src/rdmaxcel.cpp");

    // Try ROCm first, fall back to CUDA
    let (is_rocm, compute_home, compute_lib_names) = if let Ok(rocm_home) = build_utils::validate_rocm_installation() {
        println!("cargo::warning=Using HIP/ROCm from {}", rocm_home);
        // Link both HIP and HSA runtime for dmabuf support
        (true, rocm_home, vec!["amdhip64", "hsa-runtime64"])
    } else if let Ok(cuda_home) = build_utils::validate_cuda_installation() {
        println!("cargo::warning=Using CUDA from {}", cuda_home);
        (false, cuda_home, vec!["cuda", "cudart"])
    } else {
        eprintln!("Error: Neither CUDA nor ROCm installation found!");
        build_utils::print_cuda_error_help();
        build_utils::print_rocm_error_help();
        std::process::exit(1);
    };

    // Get the directory of the current crate
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| {
        // For buck2 run, we know the package is in fbcode/monarch/rdmaxcel-sys
        // Get the fbsource directory from the current directory path
        let current_dir = std::env::current_dir().expect("Failed to get current directory");
        let current_path = current_dir.to_string_lossy();

        // Find the fbsource part of the path
        if let Some(fbsource_pos) = current_path.find("fbsource") {
            let fbsource_path = &current_path[..fbsource_pos + "fbsource".len()];
            format!("{}/fbcode/monarch/rdmaxcel-sys", fbsource_path)
        } else {
            // If we can't find fbsource in the path, just use the current directory
            format!("{}/src", current_dir.to_string_lossy())
        }
    });

    // Create the absolute path to the header file
    let header_path = format!("{}/src/rdmaxcel.h", manifest_dir);

    // Check if the header file exists
    if !Path::new(&header_path).exists() {
        panic!("Header file not found at {}", header_path);
    }

    // Start building the bindgen configuration
    let mut builder = bindgen::Builder::default()
        // The input header we would like to generate bindings for
        .header(&header_path)
        .clang_arg("-x")
        .clang_arg("c++")
        .clang_arg("-std=gnu++20")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        // Allow the specified functions, types, and variables
        .allowlist_function("ibv_.*")
        .allowlist_function("mlx5dv_.*")
        .allowlist_function("mlx5_wqe_.*")
        .allowlist_function("create_qp")
        .allowlist_function("create_mlx5dv_.*")
        .allowlist_function("register_cuda_memory")
        .allowlist_function("db_ring")
        .allowlist_function("cqe_poll")
        .allowlist_function("send_wqe")
        .allowlist_function("recv_wqe")
        .allowlist_function("launch_db_ring")
        .allowlist_function("launch_cqe_poll")
        .allowlist_function("launch_send_wqe")
        .allowlist_function("launch_recv_wqe")
        .allowlist_function("rdma_get_active_segment_count")
        .allowlist_function("rdma_get_all_segment_info")
        .allowlist_function("pt_cuda_allocator_compatibility")
        .allowlist_function("register_segments")
        .allowlist_function("deregister_segments")
        .allowlist_function("register_dmabuf_buffer")
        .allowlist_type("ibv_.*")
        .allowlist_type("mlx5dv_.*")
        .allowlist_type("mlx5_wqe_.*")
        .allowlist_type("cqe_poll_result_t")
        .allowlist_type("wqe_params_t")
        .allowlist_type("cqe_poll_params_t")
        .allowlist_type("rdma_segment_info_t")
        .allowlist_var("MLX5_.*")
        .allowlist_var("IBV_.*")
        .allowlist_var("RDMA_QP_TYPE_.*")
        // Block specific types that are manually defined in lib.rs
        .blocklist_type("ibv_wc")
        .blocklist_type("mlx5_wqe_ctrl_seg")
        // Apply the same bindgen flags as in the BUCK file
        .bitfield_enum("ibv_access_flags")
        .bitfield_enum("ibv_qp_attr_mask")
        .bitfield_enum("ibv_wc_flags")
        .bitfield_enum("ibv_send_flags")
        .bitfield_enum("ibv_port_cap_flags")
        .constified_enum_module("ibv_qp_type")
        .constified_enum_module("ibv_qp_state")
        .constified_enum_module("ibv_port_state")
        .constified_enum_module("ibv_wc_opcode")
        .constified_enum_module("ibv_wr_opcode")
        .constified_enum_module("ibv_wc_status")
        .derive_default(true)
        .prepend_enum_name(false);

    // Add compute (CUDA/ROCm) include path (we already validated it exists)
    let compute_include_path = format!("{}/include", compute_home);
    println!("cargo:rustc-env=CUDA_INCLUDE_PATH={}", compute_include_path);
    builder = builder.clang_arg(format!("-I{}", compute_include_path));

    // Define HIP platform macros when using ROCm
    if is_rocm {
        builder = builder
            .clang_arg("-D__HIP_PLATFORM_AMD__=1")
            .clang_arg("-DUSE_ROCM=1");
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
        println!("cargo:rustc-link-search=native={}", lib_dir);
        println!("cargo:metadata=LIB_PATH={}", lib_dir);
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
    for lib_name in &compute_lib_names {
        println!("cargo:rustc-link-lib={}", lib_name);
    }

    // Link PyTorch C++ libraries for c10 symbols
    let use_pytorch_apis = build_utils::get_env_var_with_rerun("TORCH_SYS_USE_PYTORCH_APIS")
        .unwrap_or_else(|_| "1".to_owned());
    if use_pytorch_apis == "1" {
        // Try to get PyTorch library directory
        let python_interpreter = std::path::PathBuf::from("python3");
        if let Ok(output) = std::process::Command::new(&python_interpreter)
            .arg("-c")
            .arg(build_utils::PYTHON_PRINT_PYTORCH_DETAILS)
            .output()
        {
            for line in String::from_utf8_lossy(&output.stdout).lines() {
                if let Some(path) = line.strip_prefix("LIBTORCH_LIB: ") {
                    println!("cargo:rustc-link-search=native={}", path);
                    break;
                }
            }
        }
        // Link core PyTorch libraries needed for C10 symbols
        println!("cargo:rustc-link-lib=torch_cpu");
        println!("cargo:rustc-link-lib=torch");
        println!("cargo:rustc-link-lib=c10");
    }

    // Generate bindings
    let bindings = builder.generate().expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file
    match env::var("OUT_DIR") {
        Ok(out_dir) => {
            // Export OUT_DIR so dependent crates can find our compiled libraries
            println!("cargo:out_dir={}", out_dir);

            let out_path = PathBuf::from(&out_dir);
            match bindings.write_to_file(out_path.join("bindings.rs")) {
                Ok(_) => {
                    println!("cargo:rustc-cfg=cargo");
                    println!("cargo:rustc-check-cfg=cfg(cargo)");
                }
                Err(e) => eprintln!("Warning: Couldn't write bindings: {}", e),
            }

            // Compile the C source file
            let c_source_path = format!("{}/src/rdmaxcel.c", manifest_dir);
            if Path::new(&c_source_path).exists() {
                let mut build = cc::Build::new();
                build
                    .file(&c_source_path)
                    .include(format!("{}/src", manifest_dir))
                    .flag("-fPIC");

                // Add compute include paths - reuse the paths we already found for bindgen
                build.include(&compute_include_path);

                // Define HIP platform macros when using ROCm
                if is_rocm {
                    build.define("__HIP_PLATFORM_AMD__", "1");
                    build.define("USE_ROCM", "1");
                }

                build.compile("rdmaxcel");
            } else {
                panic!("C source file not found at {}", c_source_path);
            }

            // Compile the C++ source file for CUDA allocator compatibility
            let cpp_source_path = format!("{}/src/rdmaxcel.cpp", manifest_dir);
            if Path::new(&cpp_source_path).exists() {
                let mut libtorch_include_dirs: Vec<PathBuf> = vec![];

                // Use the same approach as torch-sys: Python discovery first, env vars as fallback
                let use_pytorch_apis =
                    build_utils::get_env_var_with_rerun("TORCH_SYS_USE_PYTORCH_APIS")
                        .unwrap_or_else(|_| "1".to_owned());

                if use_pytorch_apis == "1" {
                    // Use Python to get PyTorch include paths (same as torch-sys)
                    let python_interpreter = PathBuf::from("python3");
                    let output = std::process::Command::new(&python_interpreter)
                        .arg("-c")
                        .arg(build_utils::PYTHON_PRINT_PYTORCH_DETAILS)
                        .output()
                        .unwrap_or_else(|_| panic!("error running {python_interpreter:?}"));

                    for line in String::from_utf8_lossy(&output.stdout).lines() {
                        if let Some(path) = line.strip_prefix("LIBTORCH_INCLUDE: ") {
                            libtorch_include_dirs.push(PathBuf::from(path));
                        }
                    }
                } else {
                    // Use environment variables (fallback approach)
                    libtorch_include_dirs.extend(
                        build_utils::get_env_var_with_rerun("LIBTORCH_INCLUDE")
                            .unwrap_or_default()
                            .split(':')
                            .filter(|s| !s.is_empty())
                            .map(PathBuf::from),
                    );
                }

                let mut cpp_build = cc::Build::new();
                cpp_build
                    .file(&cpp_source_path)
                    .include(format!("{}/src", manifest_dir))
                    .flag("-fPIC")
                    .cpp(true)
                    .flag("-std=gnu++20")
                    .define("PYTORCH_C10_DRIVER_API_SUPPORTED", "1");

                // Add compute include paths
                cpp_build.include(&compute_include_path);

                // Define HIP platform macros when using ROCm
                if is_rocm {
                    cpp_build.define("__HIP_PLATFORM_AMD__", "1");
                    cpp_build.define("USE_ROCM", "1");
                }

                // Add PyTorch/C10 include paths
                for include_dir in &libtorch_include_dirs {
                    cpp_build.include(include_dir);
                }

                // Add Python include path if available
                if let Some(include_dir) = &python_config.include_dir {
                    cpp_build.include(include_dir);
                }

                cpp_build.compile("rdmaxcel_cpp");
            } else {
                panic!("C++ source file not found at {}", cpp_source_path);
            }
            // Compile the CUDA/HIP source file
            let cuda_source_path = format!("{}/src/rdmaxcel.cu", manifest_dir);
            if Path::new(&cuda_source_path).exists() {
                // Use the appropriate compiler (hipcc for ROCm, nvcc for CUDA)
                let (compiler_path, compiler_name) = if is_rocm {
                    (format!("{}/bin/hipcc", compute_home), "hipcc")
                } else {
                    (format!("{}/bin/nvcc", compute_home), "nvcc")
                };

                // Set up fixed output directory - use a predictable path instead of dynamic OUT_DIR
                let cuda_build_dir = format!("{}/target/cuda_build", manifest_dir);
                std::fs::create_dir_all(&cuda_build_dir)
                    .expect("Failed to create CUDA build directory");

                let cuda_obj_path = format!("{}/rdmaxcel_cuda.o", cuda_build_dir);
                let cuda_lib_path = format!("{}/librdmaxcel_cuda.a", cuda_build_dir);

                // Compile with the appropriate compiler (hipcc or nvcc)
                let compiler_output = if is_rocm {
                    // HIP compilation with hipcc
                    std::process::Command::new(&compiler_path)
                        .args(&[
                            "-c",
                            &cuda_source_path,
                            "-o",
                            &cuda_obj_path,
                            "-fPIC",
                            "-std=c++20",
                            "-D__HIP_PLATFORM_AMD__=1",
                            "-DUSE_ROCM=1",
                            &format!("-I{}", compute_include_path),
                            &format!("-I{}/src", manifest_dir),
                            &format!("-I/usr/include"),
                            &format!("-I/usr/include/infiniband"),
                        ])
                        .output()
                } else {
                    // CUDA compilation with nvcc
                    std::process::Command::new(&compiler_path)
                        .args(&[
                            "-c",
                            &cuda_source_path,
                            "-o",
                            &cuda_obj_path,
                            "--compiler-options",
                            "-fPIC",
                            "-std=c++20",
                            "--expt-extended-lambda",
                            "-Xcompiler",
                            "-fPIC",
                            &format!("-I{}", compute_include_path),
                            &format!("-I{}/src", manifest_dir),
                            &format!("-I/usr/include"),
                            &format!("-I/usr/include/infiniband"),
                        ])
                        .output()
                };

                match compiler_output {
                    Ok(output) => {
                        if !output.status.success() {
                            eprintln!("{} stderr: {}", compiler_name, String::from_utf8_lossy(&output.stderr));
                            eprintln!("{} stdout: {}", compiler_name, String::from_utf8_lossy(&output.stdout));
                            panic!("Failed to compile CUDA/HIP source with {}", compiler_name);
                        }
                        println!("cargo:rerun-if-changed={}", cuda_source_path);
                    }
                    Err(e) => {
                        eprintln!("Failed to run {}: {}", compiler_name, e);
                        panic!("{} not found or failed to execute", compiler_name);
                    }
                }

                // Create static library from the compiled CUDA object
                let ar_output = std::process::Command::new("ar")
                    .args(&["rcs", &cuda_lib_path, &cuda_obj_path])
                    .output();

                match ar_output {
                    Ok(output) => {
                        if !output.status.success() {
                            eprintln!("ar stderr: {}", String::from_utf8_lossy(&output.stderr));
                            panic!("Failed to create CUDA static library with ar");
                        }
                        // Emit metadata so dependent crates can find this library
                        println!("cargo:rustc-link-lib=static=rdmaxcel_cuda");
                        println!("cargo:rustc-link-search=native={}", cuda_build_dir);

                        // Copy the library to OUT_DIR as well for Cargo dependency mechanism
                        if let Err(e) =
                            std::fs::copy(&cuda_lib_path, format!("{}/librdmaxcel_cuda.a", out_dir))
                        {
                            eprintln!("Warning: Failed to copy CUDA library to OUT_DIR: {}", e);
                        }
                    }
                    Err(e) => {
                        eprintln!("Failed to run ar: {}", e);
                        panic!("ar not found or failed to execute");
                    }
                }
            } else {
                panic!("CUDA source file not found at {}", cuda_source_path);
            }
        }
        Err(_) => {
            println!("Note: OUT_DIR not set, skipping bindings file generation");
        }
    }
}
