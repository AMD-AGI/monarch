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

    // Check USE_ROCM environment variable to decide between CUDA and ROCm
    let use_rocm = build_utils::use_rocm();

    let (accelerator_home, accelerator_include_path, accelerator_lib_dir) = if use_rocm {
        println!("cargo:rustc-cfg=feature=\"rocm\"");
        println!("cargo:rustc-check-cfg=cfg(feature, values(\"rocm\"))");

        // Use ROCm/HIP
        let hip_config = match build_utils::discover_hip_config() {
            Ok(config) => config,
            Err(_) => {
                build_utils::print_rocm_error_help();
                std::process::exit(1);
            }
        };

        let rocm_home = hip_config.rocm_home.expect("ROCm home not found");
        let hip_include = format!("{}/include", rocm_home.display());
        let hip_lib = match build_utils::get_rocm_lib_dir() {
            Ok(dir) => dir,
            Err(_) => {
                build_utils::print_rocm_lib_error_help();
                std::process::exit(1);
            }
        };

        (rocm_home.to_string_lossy().to_string(), hip_include, hip_lib)
    } else {
        println!("cargo:rustc-cfg=feature=\"cuda\"");
        println!("cargo:rustc-check-cfg=cfg(feature, values(\"cuda\"))");

        // Use CUDA
        let cuda_home = match build_utils::validate_cuda_installation() {
            Ok(home) => home,
            Err(_) => {
                build_utils::print_cuda_error_help();
                std::process::exit(1);
            }
        };

        let cuda_include = format!("{}/include", cuda_home);
        let cuda_lib = match build_utils::get_cuda_lib_dir() {
            Ok(dir) => dir,
            Err(_) => {
                build_utils::print_cuda_lib_error_help();
                std::process::exit(1);
            }
        };

        (cuda_home, cuda_include, cuda_lib)
    };

    let hip_home = accelerator_home.clone();

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
        .allowlist_type("ibv_.*")
        .allowlist_type("mlx5dv_.*")
        .allowlist_type("mlx5_wqe_.*")
        .allowlist_type("cqe_poll_result_t")
        .allowlist_type("wqe_params_t")
        .allowlist_type("cqe_poll_params_t")
        .allowlist_type("rdma_segment_info_t")
        .allowlist_var("MLX5_.*")
        .allowlist_var("IBV_.*")
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

    // Add accelerator include path
    println!("cargo:rustc-env=HIP_INCLUDE_PATH={}", accelerator_include_path);
    builder = builder.clang_arg(format!("-I{}", accelerator_include_path));

    // Add platform-specific defines
    if use_rocm {
        builder = builder.clang_arg("-D__HIP_PLATFORM_AMD__");
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

    // Link against accelerator libraries
    println!("cargo:rustc-link-search=native={}", accelerator_lib_dir);

    if use_rocm {
        println!("cargo:rustc-link-lib=amdhip64");
    } else {
        println!("cargo:rustc-link-lib=cuda");
        println!("cargo:rustc-link-lib=cudart");
    }

    // Link PyTorch C++ libraries for c10 symbols
    let use_pytorch_apis = build_utils::get_env_var_with_rerun("TORCH_SYS_USE_PYTORCH_APIS")
        .unwrap_or_else(|_| "1".to_owned());
    if use_pytorch_apis == "1" {
        // Try to get PyTorch library directory
        // Try to find Python - check venv first, then python3, then python
        let python_paths = [
            "../.venv/bin/python",
            ".venv/bin/python",
        ];
        let mut python_interpreter = None;
        for path in &python_paths {
            if Path::new(path).exists() {
                python_interpreter = Some(std::path::PathBuf::from(path));
                break;
            }
        }
        if python_interpreter.is_none() {
            python_interpreter = Some(std::path::PathBuf::from("python3"));
        }

        if let Some(python) = python_interpreter {
            if let Ok(output) = std::process::Command::new(&python)
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

                // Add platform-specific defines and includes
                if use_rocm {
                    build.flag("-D__HIP_PLATFORM_AMD__");
                }
                build.include(&accelerator_include_path);

                build.compile("rdmaxcel");
            } else {
                panic!("C source file not found at {}", c_source_path);
            }

            // Check if PyTorch APIs are needed before compiling C++ file
            let use_pytorch_apis =
                build_utils::get_env_var_with_rerun("TORCH_SYS_USE_PYTORCH_APIS")
                    .unwrap_or_else(|_| "1".to_owned());

            // Compile the C++ source file for CUDA allocator compatibility (only if PyTorch is enabled)
            let cpp_source_path = format!("{}/src/rdmaxcel.cpp", manifest_dir);
            if use_pytorch_apis == "1" && Path::new(&cpp_source_path).exists() {
                let mut libtorch_include_dirs: Vec<PathBuf> = vec![];

                // Use Python to get PyTorch include paths (same as torch-sys)
                // Try to find Python - check venv first, then python3, then python
                let python_paths = [
                    "../.venv/bin/python",
                    ".venv/bin/python",
                ];
                let mut python_interpreter = None;
                for path in &python_paths {
                    if Path::new(path).exists() {
                        python_interpreter = Some(PathBuf::from(path));
                        break;
                    }
                }
                if python_interpreter.is_none() {
                    python_interpreter = Some(PathBuf::from("python3"));
                }

                let python_interpreter = python_interpreter.unwrap();
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

                let mut cpp_build = cc::Build::new();
                cpp_build
                    .file(&cpp_source_path)
                    .include(format!("{}/src", manifest_dir))
                    .flag("-fPIC")
                    .cpp(true)
                    .flag("-std=gnu++20")
                    .define("PYTORCH_C10_DRIVER_API_SUPPORTED", "1");

                // Add platform-specific defines and includes
                if use_rocm {
                    cpp_build.flag("-D__HIP_PLATFORM_AMD__");
                }
                cpp_build.include(&accelerator_include_path);


                // Add PyTorch/C10 include paths
                for include_dir in &libtorch_include_dirs {
                    cpp_build.include(include_dir);
                }

                // Add Python include path if available
                if let Some(include_dir) = &python_config.include_dir {
                    cpp_build.include(include_dir);
                }

                cpp_build.compile("rdmaxcel_cpp");
            } else if use_pytorch_apis == "1" {
                // PyTorch is enabled but C++ source file doesn't exist
                panic!("C++ source file not found at {}", cpp_source_path);
            }
            // If PyTorch is disabled, skip C++ compilation entirely
            // Compile the HIP or CUDA source file
            let (source_extension, compiler_name, lib_name) = if use_rocm {
                ("hip", "hipcc", "rdmaxcel_hip")
            } else {
                ("cu", "nvcc", "rdmaxcel_cuda")
            };

            let source_path = format!("{}/src/rdmaxcel.{}", manifest_dir, source_extension);
            if Path::new(&source_path).exists() {
                let compiler_path = format!("{}/bin/{}", accelerator_home, compiler_name);

                // Set up fixed output directory
                let build_dir = format!("{}/target/{}_build", manifest_dir, source_extension);
                std::fs::create_dir_all(&build_dir)
                    .expect(&format!("Failed to create {} build directory", source_extension));

                let obj_path = format!("{}/{}.o", build_dir, lib_name);
                let lib_path = format!("{}/lib{}.a", build_dir, lib_name);

                // Compile the source file
                let mut compile_cmd = std::process::Command::new(&compiler_path);
                compile_cmd
                    .args(&[
                        "-c",
                        &source_path,
                        "-o",
                        &obj_path,
                        "-fPIC",
                        "-std=c++20",
                    ]);

                if use_rocm {
                    // HIP-specific args
                    compile_cmd.args(&[
                        "-Xcompiler",
                        "-fPIC",
                    ]);
                } else {
                    // CUDA-specific args
                    compile_cmd.args(&[
                        "--compiler-options",
                        "-fPIC",
                        "--expt-extended-lambda",
                    ]);
                }

                compile_cmd.args(&[
                    &format!("-I{}", accelerator_include_path),
                    &format!("-I{}/src", manifest_dir),
                    &format!("-I/usr/include"),
                    &format!("-I/usr/include/infiniband"),
                ]);

                let compile_output = compile_cmd.output();

                match compile_output {
                    Ok(output) => {
                        if !output.status.success() {
                            eprintln!("{} stderr: {}", compiler_name, String::from_utf8_lossy(&output.stderr));
                            eprintln!("{} stdout: {}", compiler_name, String::from_utf8_lossy(&output.stdout));
                            panic!("Failed to compile {} source with {}", source_extension, compiler_name);
                        }
                        println!("cargo:rerun-if-changed={}", source_path);
                    }
                    Err(e) => {
                        eprintln!("Failed to run {}: {}", compiler_name, e);
                        panic!("{} not found or failed to execute", compiler_name);
                    }
                }

                // Create static library from the compiled object
                let ar_output = std::process::Command::new("ar")
                    .args(&["rcs", &lib_path, &obj_path])
                    .output();

                match ar_output {
                    Ok(output) => {
                        if !output.status.success() {
                            eprintln!("ar stderr: {}", String::from_utf8_lossy(&output.stderr));
                            panic!("Failed to create static library with ar");
                        }
                        // Emit metadata so dependent crates can find this library
                        println!("cargo:rustc-link-lib=static={}", lib_name);
                        println!("cargo:rustc-link-search=native={}", build_dir);

                        // Copy the library to OUT_DIR as well for Cargo dependency mechanism
                        if let Err(e) =
                            std::fs::copy(&lib_path, format!("{}/lib{}.a", out_dir, lib_name))
                        {
                            eprintln!("Warning: Failed to copy library to OUT_DIR: {}", e);
                        }
                    }
                    Err(e) => {
                        eprintln!("Failed to run ar: {}", e);
                        panic!("ar not found or failed to execute");
                    }
                }
            } else {
                panic!("{} source file not found at {}", source_extension.to_uppercase(), source_path);
            }
        }
        Err(_) => {
            println!("Note: OUT_DIR not set, skipping bindings file generation");
        }
    }
}
