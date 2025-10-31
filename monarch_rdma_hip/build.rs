/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#[cfg(target_os = "macos")]
fn main() {}

#[cfg(not(target_os = "macos"))]
fn main() {
    // Check USE_ROCM environment variable to decide between CUDA and ROCm
    let use_rocm = build_utils::use_rocm();

    if use_rocm {
        println!("cargo:rustc-cfg=feature=\"rocm\"");
        println!("cargo:rustc-check-cfg=cfg(feature, values(\"rocm\"))");
    } else {
        println!("cargo:rustc-cfg=feature=\"cuda\"");
        println!("cargo:rustc-check-cfg=cfg(feature, values(\"cuda\"))");
    }

    let (accelerator_home, accelerator_lib_dir) = if use_rocm {
        // Use ROCm/HIP
        let hip_config = match build_utils::discover_hip_config() {
            Ok(config) => config,
            Err(_) => {
                build_utils::print_rocm_error_help();
                std::process::exit(1);
            }
        };

        let rocm_home = hip_config.rocm_home.expect("ROCm home not found");
        let hip_lib = match build_utils::get_rocm_lib_dir() {
            Ok(dir) => dir,
            Err(_) => {
                build_utils::print_rocm_lib_error_help();
                std::process::exit(1);
            }
        };

        (rocm_home.to_string_lossy().to_string(), hip_lib)
    } else {
        // Use CUDA
        let cuda_home = match build_utils::validate_cuda_installation() {
            Ok(home) => home,
            Err(_) => {
                build_utils::print_cuda_error_help();
                std::process::exit(1);
            }
        };

        let cuda_lib = match build_utils::get_cuda_lib_dir() {
            Ok(dir) => dir,
            Err(_) => {
                build_utils::print_cuda_lib_error_help();
                std::process::exit(1);
            }
        };

        (cuda_home, cuda_lib)
    };

    let hip_home = accelerator_home.clone();


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

    if let Some(lib_dir) = &python_config.lib_dir {
        println!("cargo:rustc-link-search=native={}", lib_dir);
        // Set cargo metadata to inform dependent binaries about how to set their
        // RPATH (see controller/build.rs for an example).
        println!("cargo:metadata=LIB_PATH={}", lib_dir);
    }

    // Get CUDA library directory and emit link directives
    // let cuda_lib_dir = match build_utils::get_cuda_lib_dir() {
    //     Ok(dir) => dir,
    //     Err(_) => {
    //         build_utils::print_cuda_lib_error_help();
    //         std::process::exit(1);
    //     }
    // };

    println!("cargo:rustc-link-search=native={}", accelerator_lib_dir);

    // Link against accelerator libraries
    if use_rocm {
        println!("cargo:rustc-link-lib=amdhip64");
    } else {
        println!("cargo:rustc-link-lib=cuda");
        println!("cargo:rustc-link-lib=cudart");
    }

    // Link against the ibverbs and mlx5 libraries (used by rdmaxcel-sys)
    println!("cargo:rustc-link-lib=ibverbs");
    println!("cargo:rustc-link-lib=mlx5");

    // Link PyTorch libraries needed for C10 symbols used by rdmaxcel-sys
    let use_pytorch_apis = build_utils::get_env_var_with_rerun("TORCH_SYS_USE_PYTORCH_APIS")
        .unwrap_or_else(|_| "1".to_owned());
    if use_pytorch_apis == "1" {
        // Get PyTorch library directory using build_utils
        let python_interpreter = std::path::PathBuf::from("python");
        if let Ok(output) = std::process::Command::new(&python_interpreter)
            .arg("-c")
            .arg(build_utils::PYTHON_PRINT_PYTORCH_DETAILS)
            .output()
        {
            if output.status.success() {
                for line in String::from_utf8_lossy(&output.stdout).lines() {
                    if let Some(path) = line.strip_prefix("LIBTORCH_LIB: ") {
                        // Add library search path
                        println!("cargo:rustc-link-search=native={}", path);
                        // Set rpath so runtime linker can find the libraries
                        println!("cargo::rustc-link-arg=-Wl,-rpath,{}", path);
                    }
                }
            }
        }

        // Link core PyTorch libraries needed for C10 symbols
        println!("cargo:rustc-link-lib=torch_cpu");
        println!("cargo:rustc-link-lib=torch");
        println!("cargo:rustc-link-lib=c10");

        if use_rocm {
            println!("cargo:rustc-link-lib=c10_hip");
        } else {
            println!("cargo:rustc-link-lib=c10_cuda");
        }
    } else {
        // Fallback to torch-sys links metadata if available
        if let Ok(torch_lib_path) = std::env::var("DEP_TORCH_LIB_PATH") {
            println!("cargo::rustc-link-arg=-Wl,-rpath,{}", torch_lib_path);
        }
    }

    // Set rpath for NCCL libraries if available
    if let Ok(nccl_lib_path) = std::env::var("DEP_NCCL_LIB_PATH") {
        println!("cargo::rustc-link-arg=-Wl,-rpath,{}", nccl_lib_path);
    }

    // Disable new dtags, as conda envs generally use `RPATH` over `RUNPATH`
    println!("cargo::rustc-link-arg=-Wl,--disable-new-dtags");

    // Link the static libraries from rdmaxcel-sys or rdmaxcel-sys-hip
    // Try the Cargo dependency mechanism first, then fall back to fixed paths
    let dep_out_dir_var = if use_rocm {
        "DEP_RDMAXCEL_SYS_HIP_OUT_DIR"
    } else {
        "DEP_RDMAXCEL_SYS_OUT_DIR"
    };

    if let Ok(rdmaxcel_out_dir) = std::env::var(dep_out_dir_var) {
        println!("cargo:rustc-link-search=native={}", rdmaxcel_out_dir);
        println!("cargo:rustc-link-lib=static=rdmaxcel");
        println!("cargo:rustc-link-lib=static=rdmaxcel_cpp");

        if use_rocm {
            println!("cargo:rustc-link-lib=static=rdmaxcel_hip");
        } else {
            println!("cargo:rustc-link-lib=static=rdmaxcel_cuda");
        }
    } else {
        eprintln!("Warning: {} not found. Using fallback paths.", dep_out_dir_var);

        // Use relative paths to the known locations
        let (rdmaxcel_dir, build_subdir, lib_name) = if use_rocm {
            ("../rdmaxcel-sys-hip", "hip_build", "rdmaxcel_hip")
        } else {
            ("../rdmaxcel-sys", "cuda_build", "rdmaxcel_cuda")
        };

        let accelerator_build_dir = format!("{}/target/{}", rdmaxcel_dir, build_subdir);
        println!("cargo:rustc-link-search=native={}", accelerator_build_dir);
        println!("cargo:rustc-link-lib=static={}", lib_name);

        // Find the most recent rdmaxcel-sys build directory for C/C++ libraries
        let monarch_target_dir = "../target/debug/build";
        if let Ok(entries) = std::fs::read_dir(monarch_target_dir) {
            let search_prefix = if use_rocm {
                "rdmaxcel-sys-hip-"
            } else {
                "rdmaxcel-sys-"
            };

            let mut rdmaxcel_dirs: Vec<_> = entries
                .filter_map(|entry| entry.ok())
                .filter(|entry| {
                    let name = entry.file_name().to_string_lossy().to_string();
                    name.starts_with(search_prefix) && !name.contains("hip-") // Avoid matching hip when looking for base sys
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
                eprintln!("Warning: No {} build directories found", search_prefix);
            }
        }
    }

    // Set build configuration flags
    println!("cargo::rustc-cfg=cargo");
    println!("cargo::rustc-check-cfg=cfg(cargo)");
}
