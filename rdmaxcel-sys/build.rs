/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

#[cfg(target_os = "macos")]
fn main() {}

/// Helper function to check if a command exists
fn command_exists(cmd: &str) -> bool {
    Command::new("which")
        .arg(cmd)
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

/// Helper function to run hipify (either hipify_torch or hipify-perl).
fn run_hipify(input: &Path, output: &Path, use_hipify_torch: bool) {
    // Only run if the output doesn't exist or the input is newer
    if output.exists() {
        if let (Ok(in_meta), Ok(out_meta)) = (input.metadata(), output.metadata()) {
            if let (Ok(in_time), Ok(out_time)) = (in_meta.modified(), out_meta.modified()) {
                if in_time < out_time {
                    return; // Output is up-to-date
                }
            }
        }
    }

    let hipify_cmd = if use_hipify_torch { "hipify_torch" } else { "hipify-perl" };

    eprintln!(
        "HIPify: Running {} {} -o {}",
        hipify_cmd,
        input.display(),
        output.display()
    );

    let mut cmd = Command::new(hipify_cmd);
    
    if use_hipify_torch {
        // hipify_torch has different argument format
        cmd.arg("--output-file").arg(output);
        cmd.arg(input);
    } else {
        // hipify-perl format
        cmd.arg(input).arg("-o").arg(output);
    }

    let output_result = cmd.output();
    
    let output = output_result.unwrap_or_else(|_| {
        panic!("Failed to run hipify tool '{}'. Is it in your PATH?", hipify_cmd)
    });

    if !output.status.success() {
        eprintln!(
            "Hipify failed for {}. Stderr: {}",
            input.display(),
            String::from_utf8_lossy(&output.stderr)
        );
        eprintln!(
            "Hipify stdout: {}",
            String::from_utf8_lossy(&output.stdout)
        );
        panic!("Hipify failed to convert source file.");
    }
}

/// Post-process hipified files to fix PyTorch-specific headers and namespaces
fn fix_pytorch_hip_headers(file_path: &Path) {
    if !file_path.exists() {
        return;
    }
    
    let content = std::fs::read_to_string(file_path)
        .expect("Failed to read hipified file");
    
    // Fix PyTorch headers - from CUDA to HIP
    let fixed = content
        // Headers
        .replace("#include <c10/cuda/CUDAAllocatorConfig.h>", "#include <c10/hip/HIPAllocatorConfig.h>")
        .replace("#include <c10/cuda/CUDACachingAllocator.h>", "#include <c10/hip/HIPCachingAllocator.h>")
        // Namespaces
        .replace("c10::cuda::CUDACachingAllocator", "c10::hip::HIPCachingAllocator")
        .replace("c10::cuda::CUDAAllocatorConfig", "c10::hip::HIPAllocatorConfig")
        // CUDA Driver API to HIP equivalents
        .replace("CUdeviceptr", "hipDeviceptr_t")
        .replace("CUresult", "hipError_t")
        .replace("CUDA_SUCCESS", "hipSuccess")
        .replace("cuMemGetHandleForAddressRange", "hipMemGetHandleForAddressRange")
        .replace("CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD", "hipMemRangeHandleTypeDmaBufFd")
        .replace("cuPointerGetAttribute", "hipPointerGetAttribute")
        .replace("CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL", "HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL")
        .replace("cuDeviceGet", "hipDeviceGet")
        .replace("CUdevice", "hipDevice_t")
        .replace("cuDeviceGetAttribute", "hipDeviceGetAttribute")
        .replace("CU_DEVICE_ATTRIBUTE_PCI_BUS_ID", "hipDeviceAttributePciBusId")
        .replace("CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID", "hipDeviceAttributePciDeviceId")
        .replace("CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID", "hipDeviceAttributePciDomainId")
        // Fix static_cast to reinterpret_cast for hipDeviceptr_t (void*)
        .replace("static_cast<hipDeviceptr_t>", "reinterpret_cast<hipDeviceptr_t>");
    
    std::fs::write(file_path, fixed)
        .expect("Failed to write fixed hipified file");
}

/// Helper function to get PyTorch includes
fn get_pytorch_includes() -> Vec<PathBuf> {
    let output = std::process::Command::new("python3")
        .arg("-c")
        .arg(build_utils::PYTHON_PRINT_PYTORCH_DETAILS)
        .output()
        .expect("Failed to run python3 to get PyTorch details");

    if !output.status.success() {
        panic!(
            "Failed to get PyTorch details from python: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|line| line.strip_prefix("LIBTORCH_INCLUDE: "))
        .map(PathBuf::from)
        .collect()
}

#[cfg(not(target_os = "macos"))]
fn main() {
    println!("cargo:rerun-if-env-changed=USE_ROCM");

    let use_rocm = build_utils::use_rocm();

    // --- 1. Platform Configuration ---
    let (
        accelerator_home,
        accelerator_include_path, // This is for nvcc/hipcc
        bindgen_include_dirs,     // This is for bindgen/clang
        compiler_name,
        lib_name_suffix,
    ) = if use_rocm {
        let config = build_utils::discover_hip_config().unwrap_or_else(|_| {
            build_utils::print_rocm_error_help();
            std::process::exit(1);
        });
        let home = config.rocm_home.expect("ROCm home not found");
        let _lib_dir = build_utils::get_rocm_lib_dir().unwrap_or_else(|_| {
            build_utils::print_rocm_lib_error_help();
            std::process::exit(1);
        });
        (
            home.to_string_lossy().to_string(),
            format!("{}/include", home.display()),
            config.include_dirs, // <-- Use the full list for bindgen
            "hipcc",
            "hip",
        )
    } else {
        let home = build_utils::validate_cuda_installation().unwrap_or_else(|_| {
            build_utils::print_cuda_error_help();
            std::process::exit(1);
        });
        let config = build_utils::discover_cuda_config().unwrap(); // We know it exists
        (
            home.clone(),
            format!("{}/include", home),
            config.include_dirs, // <-- Use the full list for bindgen
            "nvcc",
            "cuda",
        )
    };

    // --- 2. Path Setup ---
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
    let build_dir = PathBuf::from(&manifest_dir)
        .join("target")
        .join(format!("{}_build", lib_name_suffix));
    std::fs::create_dir_all(&build_dir).expect("Failed to create build directory");

    let source_dir = PathBuf::from(&manifest_dir).join("src");
    let original_cu = source_dir.join("rdmaxcel.cu");
    let original_cpp = source_dir.join("rdmaxcel.cpp");
    let original_c = source_dir.join("rdmaxcel.c");
    let original_h = source_dir.join("rdmaxcel.h");

    // --- 3. Conditional HIPify File Generation ---
    let (accel_src_to_compile, cpp_src_to_compile, c_src_to_compile, header_to_parse) = if use_rocm
    {
        // Check if hipify_torch is available, otherwise fall back to hipify-perl
        let use_hipify_torch = command_exists("hipify_torch");
        
        if use_hipify_torch {
            eprintln!("Using hipify_torch for better PyTorch support");
        } else {
            eprintln!("hipify_torch not found, using hipify-perl with post-processing");
        }

        let hip_cu = build_dir.join("rdmaxcel.hip");
        let hip_cpp = build_dir.join("rdmaxcel.cpp");
        let hip_h = build_dir.join("rdmaxcel.h");
        let hip_c = build_dir.join("rdmaxcel.c");

        run_hipify(&original_cu, &hip_cu, use_hipify_torch);
        run_hipify(&original_cpp, &hip_cpp, use_hipify_torch);
        run_hipify(&original_h, &hip_h, use_hipify_torch);
        run_hipify(&original_c, &hip_c, use_hipify_torch);

        // If we used hipify-perl, post-process to fix PyTorch headers
        if !use_hipify_torch {
            eprintln!("Post-processing hipified files to fix PyTorch headers...");
            fix_pytorch_hip_headers(&hip_cpp);
            fix_pytorch_hip_headers(&hip_h);
            // .cu and .c files probably don't need PyTorch header fixes
        }

        println!("cargo:rerun-if-changed={}", original_cu.display());
        println!("cargo:rerun-if-changed={}", original_cpp.display());
        println!("cargo:rerun-if-changed={}", original_h.display());
        println!("cargo:rerun-if-changed={}", original_c.display());
        (hip_cu, hip_cpp, hip_c, hip_h)
    } else {
        println!("cargo:rerun-if-changed={}", original_cu.display());
        println!("cargo:rerun-if-changed={}", original_cpp.display());
        println!("cargo:rerun-if-changed={}", original_h.display());
        println!("cargo:rerun-if-changed={}", original_c.display());
        (
            original_cu.clone(),
            original_cpp.clone(),
            original_c.clone(),
            original_h.clone(),
        )
    };

    // --- 4. Link IB Verbs (Always) ---
    println!("cargo:rustc-link-lib=ibverbs");
    println!("cargo:rustc-link-lib=mlx5");

    // --- 5. Bindgen (Uses generated or original header) ---
    let mut builder = bindgen::Builder::default()
        .header(header_to_parse.to_string_lossy().as_ref())
        .clang_arg("-x")
        .clang_arg("c++")
        .clang_arg("-std=gnu++20")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .allowlist_function("ibv_.*")
        .allowlist_function("mlx5dv_.*")
        .allowlist_function("mlx5_wqe_.*")
        .allowlist_function("create_qp")
        .allowlist_function("create_mlx5dv_.*")
        .allowlist_function("(register_cuda_memory|register_hip_memory)")
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
        .allowlist_function("(pt_cuda_allocator_compatibility|pt_hip_allocator_compatibility)")
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
        .blocklist_type("ibv_wc")
        .blocklist_type("mlx5_wqe_ctrl_seg")
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

    for include_dir in &bindgen_include_dirs {
        builder = builder.clang_arg(format!("-I{}", include_dir.display()));
    }
    if use_rocm {
        builder = builder.clang_arg("-D__HIP_PLATFORM_AMD__");
    }

    let bindings = builder.generate().expect("Unable to generate bindings");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_dir.join("bindings.rs"))
        .expect("Couldn't write bindings!");
    println!("cargo:out_dir={}", out_dir.display()); // Export OUT_DIR

    // --- 6. Compile C, C++, and Accelerator Code ---

    // 6a. Compile C source (generated or original file, standard compiler)
    let mut c_build = cc::Build::new();
    c_build
        .file(&c_src_to_compile)
        .include(&build_dir)
        .include(&source_dir)
        .include(&accelerator_include_path) // Use the simple path for cc
        .flag("-fPIC");

    if use_rocm {
        c_build.define("__HIP_PLATFORM_AMD__", None);
    }
    c_build.compile("rdmaxcel");

    // 6b. Compile C++ source (generated or original file, standard compiler)
    let torch_includes = get_pytorch_includes();

    let mut cpp_build = cc::Build::new();
    cpp_build
        .file(&cpp_src_to_compile) // <-- Uses generated or original .cpp
        .include(&build_dir)
        .include(&source_dir)
        .include(&accelerator_include_path) // Use the simple path for cc
        .flag("-fPIC")
        .cpp(true)
        .flag("-std=gnu++20")
        .define("PYTORCH_C10_DRIVER_API_SUPPORTED", "1");

    // Add PyTorch/C10 include paths
    for include_path in &torch_includes {
        cpp_build.include(include_path);
    }

    if use_rocm {
        cpp_build.define("__HIP_PLATFORM_AMD__", None);
    }
    cpp_build.compile("rdmaxcel_cpp");

    // 6c. Compile Accelerator source (generated or original, platform compiler)
    let compiler_path = PathBuf::from(&accelerator_home)
        .join("bin")
        .join(compiler_name);
    let obj_path = build_dir.join(format!("rdmaxcel_{}.o", lib_name_suffix));
    let lib_path = build_dir.join(format!("librdmaxcel_{}.a", lib_name_suffix));

    let mut compile_cmd = Command::new(&compiler_path);
    compile_cmd
        .arg("-c")
        .arg(&accel_src_to_compile) // <-- Uses generated .hip or original .cu
        .arg("-o")
        .arg(&obj_path)
        .arg("-fPIC")
        .arg("-std=c++20")
        .arg(format!("-I{}", accelerator_include_path)) // Use simple path for hipcc
        .arg(format!("-I{}", source_dir.display()))
        .arg("-I/usr/include")
        .arg("-I/usr/include/infiniband");

    if use_rocm {
        compile_cmd.args(["-Xcompiler", "-fPIC"]);
    } else {
        compile_cmd.args(["--compiler-options", "-fPIC", "--expt-extended-lambda"]);
    }

    let compile_output = compile_cmd.output().unwrap_or_else(|e| {
        panic!("Failed to run compiler {}: {}", compiler_path.display(), e);
    });
    if !compile_output.status.success() {
        panic!(
            "Compiler failed: {} \nStderr: {}\nStdout: {}",
            compiler_path.display(),
            String::from_utf8_lossy(&compile_output.stderr),
            String::from_utf8_lossy(&compile_output.stdout)
        );
    }

    // Archive the compiled object
    let ar_output = Command::new("ar")
        .args(["rcs", &lib_path.to_string_lossy(), &obj_path.to_string_lossy()])
        .output()
        .expect("Failed to run 'ar'");

    if !ar_output.status.success() {
        panic!(
            "'ar' failed: {}",
            String::from_utf8_lossy(&ar_output.stderr)
        );
    }

    // --- 7. Final Link Directives ---
    println!("cargo:rustc-link-search=native={}", build_dir.display());
    println!("cargo:rustc-link-lib=static=rdmaxcel_{}", lib_name_suffix);
    // Copy lib to OUT_DIR so dependent crates can find it
    std::fs::copy(
        &lib_path,
        out_dir.join(format!("librdmaxcel_{}.a", lib_name_suffix)),
    )
    .expect("Failed to copy lib to OUT_DIR");
}
