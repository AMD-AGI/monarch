/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! This build script locates CUDA libraries and headers for torch-sys-cuda,
//! which provides CUDA-specific PyTorch functionality. It depends on the base
//! torch-sys crate for core PyTorch integration.

#![feature(exit_status_error)]

use std::path::PathBuf;
use std::process::Stdio;

use build_utils::*;
use cxx_build::CFG;
use pyo3_build_config::InterpreterConfig;

#[cfg(target_os = "macos")]
fn main() {}

#[cfg(not(target_os = "macos"))]
fn main() {
    // Check if we're building for ROCm
    if let Some(rocm_home) = build_utils::find_rocm_home() {
        println!("cargo::warning=Building torch-sys-cuda for ROCm using HIP bridge");
        println!("cargo::rustc-cfg=rocm_disabled");
        println!("cargo::rustc-check-cfg=cfg(rocm_disabled)");

        // Build the ROCm bridge instead of the CUDA bridge
        build_rocm_bridge(&rocm_home);
        return;
    }

    let mut libtorch_include_dirs: Vec<PathBuf> = vec![];
    let mut libtorch_lib_dir: Option<PathBuf> = None;
    let mut cxx11_abi = None;
    let mut cuda_home: Option<PathBuf> = None;
    let python_interpreter = std::env::var("PYO3_PYTHON")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("python"));

    let use_pytorch_apis = build_utils::get_env_var_with_rerun("TORCH_SYS_USE_PYTORCH_APIS")
        .unwrap_or_else(|_| "1".to_owned());

    if use_pytorch_apis == "1" {
        // We use the user's python installation of PyTorch to get the proper
        // headers/libraries for libtorch
        let output = std::process::Command::new(&python_interpreter)
            .arg("-c")
            .arg(build_utils::PYTHON_PRINT_CUDA_DETAILS)
            .stdout(Stdio::piped())
            .spawn()
            .unwrap_or_else(|_| panic!("error spawning {python_interpreter:?}"))
            .wait_with_output()
            .unwrap_or_else(|_| panic!("error waiting for {python_interpreter:?}"));
        output
            .status
            .exit_ok()
            .unwrap_or_else(|_| panic!("error running {python_interpreter:?}"));

        for line in String::from_utf8_lossy(&output.stdout).lines() {
            match line.strip_prefix("LIBTORCH_CXX11: ") {
                Some("False") => cxx11_abi = Some("0".to_owned()),
                Some("True") => cxx11_abi = Some("1".to_owned()),
                _ => {}
            };
            if let Some(path) = line.strip_prefix("LIBTORCH_INCLUDE: ") {
                libtorch_include_dirs.push(PathBuf::from(path))
            }
            if let Some(path) = line.strip_prefix("LIBTORCH_LIB: ") {
                libtorch_lib_dir = Some(PathBuf::from(path))
            }
            if let Some(path) = line.strip_prefix("CUDA_HOME: ") {
                cuda_home = Some(PathBuf::from(path));
            }
        }
    } else {
        cxx11_abi = Some(build_utils::get_env_var_with_rerun("_GLIBCXX_USE_CXX11_ABI").unwrap());
        libtorch_include_dirs.extend(
            build_utils::get_env_var_with_rerun("LIBTORCH_INCLUDE")
                .unwrap()
                .split(':')
                .map(|s| s.into()),
        );
        libtorch_lib_dir = Some(
            build_utils::get_env_var_with_rerun("LIBTORCH_LIB")
                .unwrap()
                .into(),
        );
        cuda_home = Some(
            build_utils::get_env_var_with_rerun("CUDA_HOME")
                .unwrap()
                .into(),
        );
    }
    // Support both CUDA and ROCm
    let (is_rocm, compute_home) = if let Some(rocm_home) = build_utils::find_rocm_home() {
        println!("cargo::warning=Using ROCm from {}", rocm_home);
        (true, PathBuf::from(rocm_home))
    } else if let Some(cuda_home_path) = cuda_home {
        println!("cargo::warning=Using CUDA from {}", cuda_home_path.display());
        (false, cuda_home_path)
    } else {
        panic!("Neither CUDA nor ROCm installation found!");
    };

    let mut python_include: Option<PathBuf> = None;
    let mut python_include_dir: Option<PathBuf> = None;
    // Include Python headers for compatibility with torch-sys
    let output = std::process::Command::new(&python_interpreter)
        .arg("-c")
        .arg(build_utils::PYTHON_PRINT_INCLUDE_PATH)
        .output()
        .unwrap_or_else(|_| panic!("error running {python_interpreter:?}"));
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        if let Some(path) = line.strip_prefix("PYTHON_INCLUDE: ") {
            python_include = Some(PathBuf::from(path));
        }
        if let Some(path) = line.strip_prefix("PYTHON_INCLUDE_DIR: ") {
            python_include_dir = Some(PathBuf::from(path));
        }
        if let Some(path) = line.strip_prefix("PYTHON_LIB_DIR: ") {
            println!("cargo::rustc-link-search=native={}", path);
        }
    }

    // Use PyO3's Python discovery to find the correct Python library paths
    // This is more robust than hardcoding platform-specific paths
    let mut python_lib_dir: Option<String> = None;
    match InterpreterConfig::from_interpreter(&python_interpreter) {
        Ok(python_config) => {
            // Add Python library directory to search path
            if let Some(lib_dir) = &python_config.lib_dir {
                println!("cargo::rustc-link-search=native={}", lib_dir);
                python_lib_dir = Some(lib_dir.clone());
            }

            // On some platforms, we may need to explicitly link against Python
            // PyO3 handles the complexity of determining when this is needed
            if let Some(lib_name) = python_config.lib_name {
                println!("cargo::rustc-link-lib={}", lib_name);
            }
        }
        Err(e) => {
            println!(
                "cargo::warning=Failed to get Python interpreter config: {}",
                e
            );
            println!("cargo::warning=This may cause linking issues with Python libraries");
        }
    }

    // Add CUDA/ROCm toolkit includes
    libtorch_include_dirs.push(format!("{}/include", compute_home.display()).into());

    // Prefix includes with `monarch` to maintain consistency with fbcode
    // folder structure
    CFG.include_prefix = "monarch/torch-sys-cuda";
    let _builder = cxx_build::bridge("src/bridge.rs")
        .file("src/bridge.cpp")
        .flag("-std=gnu++20")
        .includes(&libtorch_include_dirs)
        .include(python_include.unwrap())
        .include(python_include_dir.unwrap())
        // Suppress warnings, otherwise we get massive spew from libtorch
        .flag_if_supported("-w")
        .flag(&format!(
            "-Wl,-rpath={}",
            libtorch_lib_dir.clone().unwrap().display()
        ))
        .flag(&format!("-D_GLIBCXX_USE_CXX11_ABI={}", cxx11_abi.unwrap()))
        // Add HIP platform defines when using ROCm
        .define("__HIP_PLATFORM_AMD__", if is_rocm { "1" } else { "0" })
        .define("USE_ROCM", if is_rocm { "1" } else { "0" })
        .compile("torch-sys-cuda");

    // Link against the PyTorch library directory for base dependencies
    println!(
        "cargo::rustc-link-search=native={}",
        libtorch_lib_dir.clone().unwrap().display()
    );

    // Configure CUDA/ROCm-specific linking
    if is_rocm {
        println!("cargo::rustc-link-lib=torch_hip");
        println!("cargo::rustc-link-lib=c10_hip");
        println!("cargo::rustc-link-lib=amdhip64");
        println!(
            "cargo::rustc-link-search=native={}/lib",
            compute_home.display()
        );
    } else {
        println!("cargo::rustc-link-lib=torch_cuda");
        println!("cargo::rustc-link-lib=c10_cuda");
        println!("cargo::rustc-link-lib=cudart");
        println!(
            "cargo::rustc-link-search=native={}/lib64",
            compute_home.display()
        );
    }

    // Set runtime paths
    println!(
        "cargo::rustc-link-arg=-Wl,-rpath,{}",
        libtorch_lib_dir.clone().unwrap().display()
    );

    // Add Python library directory to rpath for runtime linking
    if let Some(python_lib_dir) = &python_lib_dir {
        println!("cargo::rustc-link-arg=-Wl,-rpath,{}", python_lib_dir);
    }

    // Set cargo metadata to inform dependent binaries about how to set their
    // RPATH (see monarch_tensor_worker/build.rs for an example).
    println!(
        "cargo::metadata=LIB_PATH={}",
        libtorch_lib_dir.clone().unwrap().display()
    );

    println!("cargo::rerun-if-changed=src/bridge.rs");
    println!("cargo::rerun-if-changed=src/bridge.cpp");
    println!("cargo::rerun-if-changed=src/bridge.h");
}

#[cfg(not(target_os = "macos"))]
fn build_rocm_bridge(rocm_home: &str) {
    let python_interpreter = std::env::var("PYO3_PYTHON")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("python3"));

    let use_pytorch_apis = build_utils::get_env_var_with_rerun("TORCH_SYS_USE_PYTORCH_APIS")
        .unwrap_or_else(|_| "1".to_owned());

    let mut libtorch_include_dirs: Vec<PathBuf> = vec![];
    let mut libtorch_lib_dir: Option<PathBuf> = None;
    let mut cxx11_abi = None;

    if use_pytorch_apis == "1" {
        // Get PyTorch paths from Python
        let output = std::process::Command::new(&python_interpreter)
            .arg("-c")
            .arg(build_utils::PYTHON_PRINT_CUDA_DETAILS)
            .stdout(Stdio::piped())
            .spawn()
            .unwrap_or_else(|_| panic!("error spawning {python_interpreter:?}"))
            .wait_with_output()
            .unwrap_or_else(|_| panic!("error waiting for {python_interpreter:?}"));
        output
            .status
            .exit_ok()
            .unwrap_or_else(|_| panic!("error running {python_interpreter:?}"));

        for line in String::from_utf8_lossy(&output.stdout).lines() {
            match line.strip_prefix("LIBTORCH_CXX11: ") {
                Some("False") => cxx11_abi = Some("0".to_owned()),
                Some("True") => cxx11_abi = Some("1".to_owned()),
                _ => {}
            };
            if let Some(path) = line.strip_prefix("LIBTORCH_INCLUDE: ") {
                libtorch_include_dirs.push(PathBuf::from(path))
            }
            if let Some(path) = line.strip_prefix("LIBTORCH_LIB: ") {
                libtorch_lib_dir = Some(PathBuf::from(path))
            }
        }
    } else {
        cxx11_abi = Some(build_utils::get_env_var_with_rerun("_GLIBCXX_USE_CXX11_ABI").unwrap());
        libtorch_include_dirs.extend(
            build_utils::get_env_var_with_rerun("LIBTORCH_INCLUDE")
                .unwrap()
                .split(':')
                .map(|s| s.into()),
        );
        libtorch_lib_dir = Some(
            build_utils::get_env_var_with_rerun("LIBTORCH_LIB")
                .unwrap()
                .into(),
        );
    }

    // Get Python include paths
    let mut python_include: Option<PathBuf> = None;
    let mut python_include_dir: Option<PathBuf> = None;
    let output = std::process::Command::new(&python_interpreter)
        .arg("-c")
        .arg(build_utils::PYTHON_PRINT_INCLUDE_PATH)
        .output()
        .unwrap_or_else(|_| panic!("error running {python_interpreter:?}"));
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        if let Some(path) = line.strip_prefix("PYTHON_INCLUDE: ") {
            python_include = Some(PathBuf::from(path));
        }
        if let Some(path) = line.strip_prefix("PYTHON_INCLUDE_DIR: ") {
            python_include_dir = Some(PathBuf::from(path));
        }
        if let Some(path) = line.strip_prefix("PYTHON_LIB_DIR: ") {
            println!("cargo::rustc-link-search=native={}", path);
        }
    }

    // Add Python library directory
    let mut python_lib_dir: Option<String> = None;
    match InterpreterConfig::from_interpreter(&python_interpreter) {
        Ok(python_config) => {
            if let Some(lib_dir) = &python_config.lib_dir {
                println!("cargo::rustc-link-search=native={}", lib_dir);
                python_lib_dir = Some(lib_dir.clone());
            }
            if let Some(lib_name) = python_config.lib_name {
                println!("cargo::rustc-link-lib={}", lib_name);
            }
        }
        Err(e) => {
            println!(
                "cargo::warning=Failed to get Python interpreter config: {}",
                e
            );
        }
    }

    // Add ROCm include directory
    libtorch_include_dirs.push(format!("{}/include", rocm_home).into());

    // Build the ROCm bridge using CXX
    CFG.include_prefix = "monarch/torch-sys-cuda";
    let _builder = cxx_build::bridge("src/bridge_rocm.rs")
        .file("src/bridge_rocm.cpp")
        .flag("-std=gnu++20")
        .includes(&libtorch_include_dirs)
        .include(python_include.unwrap())
        .include(python_include_dir.unwrap())
        .flag_if_supported("-w")
        .flag(&format!(
            "-Wl,-rpath={}",
            libtorch_lib_dir.clone().unwrap().display()
        ))
        .flag(&format!("-D_GLIBCXX_USE_CXX11_ABI={}", cxx11_abi.unwrap()))
        .define("__HIP_PLATFORM_AMD__", "1")
        .define("USE_ROCM", "1")
        .compile("torch-sys-cuda-rocm");

    // Link against PyTorch HIP libraries
    println!(
        "cargo::rustc-link-search=native={}",
        libtorch_lib_dir.clone().unwrap().display()
    );
    println!("cargo::rustc-link-lib=torch_hip");
    println!("cargo::rustc-link-lib=c10_hip");
    println!("cargo::rustc-link-lib=amdhip64");
    println!("cargo::rustc-link-search=native={}/lib", rocm_home);

    // Set runtime paths
    println!(
        "cargo::rustc-link-arg=-Wl,-rpath,{}",
        libtorch_lib_dir.clone().unwrap().display()
    );
    if let Some(python_lib_dir) = &python_lib_dir {
        println!("cargo::rustc-link-arg=-Wl,-rpath,{}", python_lib_dir);
    }

    // Set cargo metadata for dependent crates
    println!(
        "cargo::metadata=LIB_PATH={}",
        libtorch_lib_dir.clone().unwrap().display()
    );

    println!("cargo::rerun-if-changed=src/bridge_rocm.rs");
    println!("cargo::rerun-if-changed=src/bridge_rocm.cpp");
    println!("cargo::rerun-if-changed=src/bridge_rocm.h");
}
