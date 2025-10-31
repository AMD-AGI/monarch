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
        println!("cargo:warning=Using ROCm backend (monarch_rdma_hip)");

        // Link against ROCm/HIP libraries
        let rocm_lib_dir = match build_utils::get_rocm_lib_dir() {
            Ok(dir) => dir,
            Err(_) => {
                build_utils::print_rocm_lib_error_help();
                std::process::exit(1);
            }
        };
        println!("cargo:rustc-link-search=native={}", rocm_lib_dir);
        println!("cargo:rustc-link-lib=amdhip64");
        // Set rpath so runtime linker can find ROCm libraries
        println!("cargo::rustc-link-arg=-Wl,-rpath,{}", rocm_lib_dir);
    } else {
        println!("cargo:rustc-cfg=feature=\"cuda\"");
        println!("cargo:rustc-check-cfg=cfg(feature, values(\"cuda\"))");
        println!("cargo:warning=Using CUDA backend (monarch_rdma)");

        // Link against CUDA libraries
        let cuda_lib_dir = match build_utils::get_cuda_lib_dir() {
            Ok(dir) => dir,
            Err(_) => {
                build_utils::print_cuda_lib_error_help();
                std::process::exit(1);
            }
        };
        println!("cargo:rustc-link-search=native={}", cuda_lib_dir);
        println!("cargo:rustc-link-lib=cuda");
        println!("cargo:rustc-link-lib=cudart");
        // Set rpath so runtime linker can find CUDA libraries
        println!("cargo::rustc-link-arg=-Wl,-rpath,{}", cuda_lib_dir);
    }

    // Add library search path for InfiniBand libraries
    println!("cargo:rustc-link-search=native=/lib64");

    // Link against the ibverbs and mlx5 libraries (required by both backends)
    println!("cargo:rustc-link-lib=ibverbs");
    println!("cargo:rustc-link-lib=mlx5");

    // Add rpath for PyTorch libraries if available
    let use_pytorch_apis = build_utils::get_env_var_with_rerun("TORCH_SYS_USE_PYTORCH_APIS")
        .unwrap_or_else(|_| "1".to_owned());
    if use_pytorch_apis == "1" {
        // Try to find Python - check venv first, then python3, then python
        let python_paths = [
            "../.venv/bin/python",
            ".venv/bin/python",
            "/home/mreso/monarch/.venv/bin/python",
        ];

        let mut python_interpreter = None;
        for path in &python_paths {
            if std::path::Path::new(path).exists() {
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
                if output.status.success() {
                    for line in String::from_utf8_lossy(&output.stdout).lines() {
                        if let Some(path) = line.strip_prefix("LIBTORCH_LIB: ") {
                            println!("cargo::rustc-link-arg=-Wl,-rpath,{}", path);
                        }
                    }
                }
            }
        }
    }

    // Disable new dtags for compatibility with conda envs
    println!("cargo::rustc-link-arg=-Wl,--disable-new-dtags");
}
