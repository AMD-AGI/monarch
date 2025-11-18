/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::env;
use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;

/// Finds the python interpreter, preferring `python3` if available.
fn find_python_interpreter() -> PathBuf {
    std::env::var("PYO3_PYTHON")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            if Command::new("python3").arg("--version").output().is_ok() {
                PathBuf::from("python3")
            } else {
                PathBuf::from("python")
            }
        })
}

/// Detects ROCm version and returns (major, minor) or None if not found
fn get_rocm_version(rocm_home: &str) -> Option<(u32, u32)> {
    // Try to read ROCm version from .info/version file
    let version_file = PathBuf::from(rocm_home).join(".info").join("version");
    if let Ok(content) = fs::read_to_string(&version_file) {
        let trimmed = content.trim();
        if let Some((major_str, rest)) = trimmed.split_once('.') {
            if let Some((minor_str, _)) = rest.split_once('.') {
                if let (Ok(major), Ok(minor)) = (major_str.parse::<u32>(), minor_str.parse::<u32>())
                {
                    println!(
                        "cargo:warning=Detected ROCm version {}.{} from {}",
                        major,
                        minor,
                        version_file.display()
                    );
                    return Some((major, minor));
                }
            }
        }
    }

    // Fallback: try hipcc --version
    let hipcc_path = format!("{}/bin/hipcc", rocm_home);
    if let Ok(output) = Command::new(&hipcc_path).arg("--version").output() {
        let version_output = String::from_utf8_lossy(&output.stdout);
        // Look for version pattern like "HIP version: 6.2.41134"
        for line in version_output.lines() {
            if line.contains("HIP version:") {
                if let Some(version_part) = line.split("HIP version:").nth(1) {
                    let version_str = version_part.trim();
                    if let Some((major_str, rest)) = version_str.split_once('.') {
                        if let Some((minor_str, _)) = rest.split_once('.') {
                            if let (Ok(major), Ok(minor)) =
                                (major_str.parse::<u32>(), minor_str.parse::<u32>())
                            {
                                println!(
                                    "cargo:warning=Detected ROCm version {}.{} from hipcc",
                                    major, minor
                                );
                                return Some((major, minor));
                            }
                        }
                    }
                }
            }
        }
    }

    println!("cargo:warning=Could not detect ROCm version, assuming 6.x");
    Some((6, 0)) // Default to 6.0 if we can't detect
}

/// Post-processes hipified files for ROCm 7.0+
fn patch_hipified_files_rocm7(hip_src_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:warning=Patching hipify_torch output for ROCm 7.0+...");

    // --- Patch the C++ file ---
    let cpp_file = hip_src_dir.join("rdmaxcel_hip.cpp");
    if cpp_file.exists() {
        let content = fs::read_to_string(&cpp_file)?;

        let patched_content = content
            // Add version header
            .replace(
                "#include <hip/hip_runtime.h>",
                "#include <hip/hip_runtime.h>\n#include <hip/hip_version.h>",
            )
            // Fix PyTorch allocator namespace
            .replace(
                "c10::cuda::CUDACachingAllocator",
                "c10::hip::HIPCachingAllocator",
            )
            .replace(
                "c10::cuda::CUDAAllocatorConfig",
                "c10::hip::HIPAllocatorConfig",
            )
            // Fix nested class names that may have been partially converted
            .replace(
                "c10::hip::HIPCachingAllocator::CUDAAllocatorConfig",
                "c10::hip::HIPCachingAllocator::HIPAllocatorConfig",
            )
            .replace("CUDAAllocatorConfig::", "HIPAllocatorConfig::")
            // NOTE: We do NOT rename custom rdmaxcel functions
            // They keep their CUDA names for backward compatibility
            // Fix HIP API issues
            .replace(
                "hipDeviceAttributePciDomainId",
                "hipDeviceAttributePciDomainID",
            )
            .replace(
                "static_cast<CUdeviceptr>",
                "reinterpret_cast<hipDeviceptr_t>",
            )
            .replace(
                "static_cast<hipDeviceptr_t>",
                "reinterpret_cast<hipDeviceptr_t>",
            )
            .replace(
                "CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD",
                "hipMemRangeHandleTypeDmaBufFd",
            )
            .replace(
                "cuMemGetHandleForAddressRange",
                "hipMemGetHandleForAddressRange",
            )
            .replace("CUDA_SUCCESS", "hipSuccess")
            .replace("CUresult", "hipError_t");

        fs::write(&cpp_file, patched_content)?;
    }

    // --- Patch the Header file ---
    let header_file = hip_src_dir.join("rdmaxcel_hip.h");
    if header_file.exists() {
        let content = fs::read_to_string(&header_file)?;
        let patched_content = content
            // Only fix CUDA API types, not custom function names
            .replace("CUdeviceptr", "hipDeviceptr_t");

        fs::write(&header_file, patched_content)?;
    }

    println!("cargo:warning=Applied ROCm 7.0+ post-processing fixes to hipified files");
    Ok(())
}

/// Post-processes files for ROCm 6.x (uses HSA dmabuf instead of HIP dmabuf)
fn patch_hipified_files_rocm6(hip_src_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:warning=Patching hipify_torch output for ROCm 6.x (HSA dmabuf)...");

    // --- Patch the C++ file ---
    let cpp_file = hip_src_dir.join("rdmaxcel_hip.cpp");
    if cpp_file.exists() {
        let content = fs::read_to_string(&cpp_file)?;

        let mut patched_content = content
            // Add version and HSA headers at the top
            .replace(
                "#include <hip/hip_runtime.h>",
                "#include <hip/hip_runtime.h>\n#include <hip/hip_version.h>\n#include <hsa/hsa.h>\n#include <hsa/hsa_ext_amd.h>"
            )
            // Fix PyTorch allocator namespace: c10::cuda → c10::hip
            .replace("c10::cuda::CUDACachingAllocator", "c10::hip::HIPCachingAllocator")
            .replace("c10::cuda::CUDAAllocatorConfig", "c10::hip::HIPAllocatorConfig")
            // Fix nested class names that may have been partially converted
            .replace("c10::hip::HIPCachingAllocator::CUDAAllocatorConfig", "c10::hip::HIPCachingAllocator::HIPAllocatorConfig")
            .replace("CUDAAllocatorConfig::", "HIPAllocatorConfig::")

            // NOTE: We do NOT rename custom rdmaxcel functions like:
            // - register_cuda_memory (stays as-is)
            // - pt_cuda_allocator_compatibility (stays as-is)
            // - get_cuda_pci_address_from_ptr (stays as-is)
            // These are user-defined functions, not CUDA API calls

            // Fix HIP API attribute names
            .replace("hipDeviceAttributePciDomainId", "hipDeviceAttributePciDomainID")

            // Fix pointer casts for HIP
            .replace("static_cast<CUdeviceptr>", "reinterpret_cast<hipDeviceptr_t>")
            .replace("static_cast<hipDeviceptr_t>", "reinterpret_cast<hipDeviceptr_t>")

            // Replace CUDA types with HIP types
            .replace("CUDA_SUCCESS", "hipSuccess")
            .replace("CUdevice device", "hipDevice_t device")

            // Fix device functions
            .replace("cuDeviceGet(&device", "hipDeviceGet(&device")
            .replace("cuDeviceGetAttribute", "hipDeviceGetAttribute")
            .replace("cuPointerGetAttribute", "hipPointerGetAttribute")

            // Fix device attribute constants
            .replace("CU_DEVICE_ATTRIBUTE_PCI_BUS_ID", "hipDeviceAttributePciBusId")
            .replace("CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID", "hipDeviceAttributePciDeviceId")
            .replace("CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID", "hipDeviceAttributePciDomainID")
            .replace("CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL", "HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL")

            // Remove CUDA-specific constants
            .replace("CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD", "/* removed - using HSA dmabuf */");

        // Critical: Replace cuMemGetHandleForAddressRange with HSA dmabuf calls
        // This needs to handle the parameter reordering properly

        // First, replace the function name globally
        patched_content = patched_content.replace(
            "cuMemGetHandleForAddressRange(",
            "hsa_amd_portable_export_dmabuf(",
        );

        // Now fix the parameter ordering for hsa_amd_portable_export_dmabuf calls
        // HSA signature: hsa_amd_portable_export_dmabuf(void* ptr, size_t size, int* fd, uint64_t* flags)
        // Old CUDA: cuMemGetHandleForAddressRange(&fd, ptr, size, type, flags)
        // New HSA:  hsa_amd_portable_export_dmabuf(ptr, size, &fd, nullptr)

        // Pattern for compact_mrs function
        patched_content = patched_content.replace(
            "hsa_amd_portable_export_dmabuf(\n      &fd,\n      reinterpret_cast<hipDeviceptr_t>(start_addr),\n      total_size,\n      /* removed - using HSA dmabuf */,\n      0);",
            "hsa_amd_portable_export_dmabuf(\n      reinterpret_cast<void*>(start_addr),\n      total_size,\n      &fd,\n      nullptr);"
        );

        // Pattern for register_segments function
        patched_content = patched_content.replace(
            "hsa_amd_portable_export_dmabuf(\n            &fd,\n            reinterpret_cast<hipDeviceptr_t>(chunk_start),\n            chunk_size,\n            /* removed - using HSA dmabuf */,\n            0);",
            "hsa_amd_portable_export_dmabuf(\n            reinterpret_cast<void*>(chunk_start),\n            chunk_size,\n            &fd,\n            nullptr);"
        );

        // More generic replacements for any other patterns
        patched_content = patched_content
            .replace(
                "hsa_amd_portable_export_dmabuf(\n      &fd,",
                "hsa_amd_portable_export_dmabuf(\n      reinterpret_cast<void*>("
            )
            .replace(
                "),\n      total_size,\n      /* removed - using HSA dmabuf */,\n      0)",
                "),\n      total_size,\n      &fd,\n      nullptr)"
            )
            .replace(
                "),\n            chunk_size,\n            /* removed - using HSA dmabuf */,\n            0)",
                "),\n            chunk_size,\n            &fd,\n            nullptr)"
            );

        // Replace result types and checks
        patched_content = patched_content
            .replace("CUresult cu_result", "hsa_status_t hsa_result")
            .replace("hipError_t cu_result", "hsa_status_t hsa_result")
            .replace(
                "cu_result != hipSuccess",
                "hsa_result != HSA_STATUS_SUCCESS",
            )
            .replace("if (cu_result", "if (hsa_result");

        // Fix get_hip_pci_address_from_ptr function - handle duplicate device_ordinal
        // This regex-like replacement handles the duplicate declaration issue
        if patched_content.contains("int get_hip_pci_address_from_ptr") {
            // Replace the function body to remove duplicate declaration
            let function_pattern = "int get_hip_pci_address_from_ptr(\n    hipDeviceptr_t cuda_ptr,\n    char* pci_addr_out,\n    size_t pci_addr_size) {\n  if (!pci_addr_out || pci_addr_size < 16) {\n    return RDMAXCEL_INVALID_PARAMS;\n  }\n\n  int device_ordinal = -1;\n  int device_ordinal = -1;";
            let function_replacement = "int get_hip_pci_address_from_ptr(\n    hipDeviceptr_t cuda_ptr,\n    char* pci_addr_out,\n    size_t pci_addr_size) {\n  if (!pci_addr_out || pci_addr_size < 16) {\n    return RDMAXCEL_INVALID_PARAMS;\n  }\n\n  int device_ordinal = -1;";
            patched_content = patched_content.replace(function_pattern, function_replacement);
        }

        // Fix hipPointerGetAttribute enum usage
        patched_content = patched_content.replace(
            "hipPointerAttribute::device",
            "HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL",
        );

        fs::write(&cpp_file, patched_content)?;
    }

    // --- Patch the Header file ---
    let header_file = hip_src_dir.join("rdmaxcel_hip.h");
    if header_file.exists() {
        let content = fs::read_to_string(&header_file)?;
        let patched_content = content
            // Only fix CUDA API types, not custom function names
            .replace("CUdeviceptr", "hipDeviceptr_t");

        fs::write(&header_file, patched_content)?;
    }

    println!("cargo:warning=Applied ROCm 6.x (HSA dmabuf) post-processing fixes to hipified files");
    Ok(())
}

/// Validates that hipified output files exist
fn validate_hipified_files(hip_src_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let required_files = [
        "rdmaxcel_hip.h",
        "rdmaxcel_hip.c",
        "rdmaxcel_hip.cpp",
        "rdmaxcel.hip",
    ];

    for file_name in &required_files {
        let file_path = hip_src_dir.join(file_name);
        if !file_path.exists() {
            return Err(format!(
                "Required hipified file {} was not found in {}",
                file_name,
                hip_src_dir.display()
            )
            .into());
        }
    }

    Ok(())
}

/// Runs `hipify_torch` on the source directory.
fn hipify_sources(
    python_interpreter: &Path,
    src_dir: &Path,
    hip_src_dir: &Path,
    rocm_version: (u32, u32),
) -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "cargo:warning=Copying sources from {} to {} for in-place hipify...",
        src_dir.display(),
        hip_src_dir.display()
    );
    fs::create_dir_all(hip_src_dir)?;

    let files_to_copy = [
        "lib.rs",
        "rdmaxcel.h",
        "rdmaxcel.c",
        "rdmaxcel.cpp",
        "rdmaxcel.cu",
        "test_rdmaxcel.c",
    ];

    for file_name in files_to_copy {
        let src_file = src_dir.join(file_name);
        let dest_file = hip_src_dir.join(file_name);
        if src_file.exists() {
            fs::copy(&src_file, &dest_file)?;
            println!("cargo:rerun-if-changed={}", src_file.display());
        }
    }

    println!("cargo:warning=Running hipify_torch in-place on copied sources with --v2...");

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);
    let project_root = manifest_dir
        .parent()
        .ok_or("Failed to find project root from manifest dir")?;
    let hipify_script = project_root
        .join("deps")
        .join("hipify_torch")
        .join("hipify_cli.py");

    if !hipify_script.exists() {
        return Err(format!("hipify_cli.py not found at {}", hipify_script.display()).into());
    }
    println!("cargo:rerun-if-changed={}", hipify_script.display());

    let hipify_output = Command::new(python_interpreter)
        .arg(&hipify_script)
        .arg("--project-directory")
        .arg(hip_src_dir)
        .arg("--v2")
        .output()?;

    if !hipify_output.status.success() {
        return Err(format!(
            "hipify_cli.py failed: {}",
            String::from_utf8_lossy(&hipify_output.stderr)
        )
        .into());
    }

    // Apply version-specific patches
    let (major, _minor) = rocm_version;
    if major >= 7 {
        patch_hipified_files_rocm7(hip_src_dir)?;
    } else {
        patch_hipified_files_rocm6(hip_src_dir)?;
    }

    Ok(())
}

#[cfg(target_os = "macos")]
fn main() {}

#[cfg(not(target_os = "macos"))]
fn main() {
    println!("cargo:rustc-link-lib=ibverbs");
    println!("cargo:rustc-link-lib=mlx5");

    let (is_rocm, compute_home, compute_lib_names, rocm_version) =
        if let Ok(rocm_home) = build_utils::validate_rocm_installation() {
            let version = get_rocm_version(&rocm_home).unwrap_or((6, 0));
            println!(
                "cargo:warning=Using HIP/ROCm {} from {}",
                format!("{}.{}", version.0, version.1),
                rocm_home
            );

            // Set compile-time flag for ROCm version
            if version.0 >= 7 {
                println!("cargo:rustc-cfg=rocm_7_plus");
            } else {
                println!("cargo:rustc-cfg=rocm_6_x");
            }

            (true, rocm_home, vec!["amdhip64", "hsa-runtime64"], version)
        } else if let Ok(cuda_home) = build_utils::validate_cuda_installation() {
            println!("cargo:warning=Using CUDA from {}", cuda_home);
            (false, cuda_home, vec!["cuda", "cudart"], (0, 0))
        } else {
            eprintln!("Error: Neither CUDA nor ROCm installation found!");
            build_utils::print_cuda_error_help();
            build_utils::print_rocm_error_help();
            std::process::exit(1);
        };

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| {
        let current_dir = std::env::current_dir().expect("Failed to get current directory");
        let current_path = current_dir.to_string_lossy();
        if let Some(fbsource_pos) = current_path.find("fbsource") {
            let fbsource_path = &current_path[..fbsource_pos + "fbsource".len()];
            format!("{}/fbcode/monarch/rdmaxcel-sys", fbsource_path)
        } else {
            format!("{}/src", current_dir.to_string_lossy())
        }
    }));
    let src_dir = manifest_dir.join("src");

    let python_interpreter = find_python_interpreter();

    let compute_include_path = format!("{}/include", compute_home);
    println!("cargo:rustc-env=CUDA_INCLUDE_PATH={}", compute_include_path);

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

    let use_pytorch_apis = build_utils::get_env_var_with_rerun("TORCH_SYS_USE_PYTORCH_APIS")
        .unwrap_or_else(|_| "1".to_owned());
    if use_pytorch_apis == "1" {
        if let Ok(output) = Command::new(&python_interpreter)
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
        println!("cargo:rustc-link-lib=torch_cpu");
        println!("cargo:rustc-link-lib=torch");
        println!("cargo:rustc-link-lib=c10");
        if is_rocm {
            println!("cargo:rustc-link-lib=c10_hip");
            println!("cargo:rustc-link-lib=c10_cuda");
        } else {
            println!("cargo:rustc-link-lib=c10_cuda");
        }
    }

    match env::var("OUT_DIR") {
        Ok(out_dir) => {
            let out_path = PathBuf::from(out_dir);
            println!("cargo:out_dir={}", out_path.display());

            let (code_dir, header_path, c_source_path, cpp_source_path, cuda_source_path);

            if is_rocm {
                let hip_src_dir = out_path.join("hipified_src");

                hipify_sources(&python_interpreter, &src_dir, &hip_src_dir, rocm_version)
                    .expect("Failed to hipify sources");

                validate_hipified_files(&hip_src_dir).expect("Hipified files validation failed");

                code_dir = hip_src_dir.clone();
                header_path = hip_src_dir.join("rdmaxcel_hip.h");
                c_source_path = hip_src_dir.join("rdmaxcel_hip.c");
                cpp_source_path = hip_src_dir.join("rdmaxcel_hip.cpp");
                cuda_source_path = hip_src_dir.join("rdmaxcel.hip");
            } else {
                println!(
                    "cargo:rerun-if-changed={}/src/rdmaxcel.h",
                    manifest_dir.display()
                );
                println!(
                    "cargo:rerun-if-changed={}/src/rdmaxcel.c",
                    manifest_dir.display()
                );
                println!(
                    "cargo:rerun-if-changed={}/src/rdmaxcel.cpp",
                    manifest_dir.display()
                );
                println!(
                    "cargo:rerun-if-changed={}/src/rdmaxcel.cu",
                    manifest_dir.display()
                );

                code_dir = src_dir.clone();
                header_path = src_dir.join("rdmaxcel.h");
                c_source_path = src_dir.join("rdmaxcel.c");
                cpp_source_path = src_dir.join("rdmaxcel.cpp");
                cuda_source_path = src_dir.join("rdmaxcel.cu");
            }

            if !header_path.exists() {
                panic!("Header file not found at {}", header_path.display());
            }

            let mut builder = bindgen::Builder::default()
                .header(header_path.to_string_lossy())
                .clang_arg("-x")
                .clang_arg("c++")
                .clang_arg("-std=gnu++20")
                .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
                .allowlist_function("ibv_.*")
                .allowlist_function("mlx5dv_.*")
                .allowlist_function("mlx5_wqe_.*")
                .allowlist_function("create_qp")
                .allowlist_function("create_mlx5dv_.*")
                .allowlist_function("register_cuda_memory")
                .allowlist_function("register_hip_memory")
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
                .allowlist_function("pt_hip_allocator_compatibility")
                .allowlist_function("register_segments")
                .allowlist_function("deregister_segments")
                .allowlist_function("register_dmabuf_buffer")
                .allowlist_function("get_hip_pci_address_from_ptr")
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

            builder = builder.clang_arg(format!("-I{}", compute_include_path));

            if is_rocm {
                builder = builder
                    .clang_arg("-D__HIP_PLATFORM_AMD__=1")
                    .clang_arg("-DUSE_ROCM=1");

                // Add version-specific defines
                if rocm_version.0 >= 7 {
                    builder = builder.clang_arg("-DROCM_7_PLUS=1");
                } else {
                    builder = builder.clang_arg("-DROCM_6_X=1");
                }
            }

            if let Some(include_dir) = &python_config.include_dir {
                builder = builder.clang_arg(format!("-I{}", include_dir));
            }

            let bindings = builder.generate().expect("Unable to generate bindings");
            bindings
                .write_to_file(out_path.join("bindings.rs"))
                .expect("Couldn't write bindings");

            println!("cargo:rustc-cfg=cargo");
            println!("cargo:rustc-check-cfg=cfg(cargo)");

            if c_source_path.exists() {
                let mut build = cc::Build::new();
                build.file(&c_source_path).include(&code_dir).flag("-fPIC");
                build.include(&compute_include_path);
                if is_rocm {
                    build.define("__HIP_PLATFORM_AMD__", "1");
                    build.define("USE_ROCM", "1");
                    if rocm_version.0 >= 7 {
                        build.define("ROCM_7_PLUS", "1");
                    } else {
                        build.define("ROCM_6_X", "1");
                    }
                }
                build.compile("rdmaxcel");
            } else {
                println!(
                    "cargo:warning=C source file {} not found, skipping C compilation.",
                    c_source_path.display()
                );
            }

            if cpp_source_path.exists() {
                let mut libtorch_include_dirs: Vec<PathBuf> = vec![];

                if use_pytorch_apis == "1" {
                    let output = Command::new(&python_interpreter)
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
                    .include(&code_dir)
                    .flag("-fPIC")
                    .cpp(true)
                    .flag("-std=gnu++20")
                    .flag("-Wno-unused-parameter")
                    .define("PYTORCH_C10_DRIVER_API_SUPPORTED", "1");

                cpp_build.include(&compute_include_path);
                if is_rocm {
                    cpp_build.define("__HIP_PLATFORM_AMD__", "1");
                    cpp_build.define("USE_ROCM", "1");
                    if rocm_version.0 >= 7 {
                        cpp_build.define("ROCM_7_PLUS", "1");
                    } else {
                        cpp_build.define("ROCM_6_X", "1");
                    }
                }
                for include_dir in &libtorch_include_dirs {
                    cpp_build.include(include_dir);
                }
                if let Some(include_dir) = &python_config.include_dir {
                    cpp_build.include(include_dir);
                }
                cpp_build.compile("rdmaxcel_cpp");
            } else {
                println!(
                    "cargo:warning=C++ source file {} not found, skipping C++ compilation.",
                    cpp_source_path.display()
                );
            }

            if cuda_source_path.exists() {
                let (compiler_path, compiler_name) = if is_rocm {
                    (format!("{}/bin/hipcc", compute_home), "hipcc")
                } else {
                    (format!("{}/bin/nvcc", compute_home), "nvcc")
                };

                let cuda_build_dir = format!("{}/target/cuda_build", manifest_dir.display());
                std::fs::create_dir_all(&cuda_build_dir)
                    .expect("Failed to create CUDA build directory");
                let cuda_obj_path = format!("{}/rdmaxcel_cuda.o", cuda_build_dir);
                let cuda_lib_path = format!("{}/librdmaxcel_cuda.a", cuda_build_dir);

                let compiler_output = if is_rocm {
                    let mut cmd = Command::new(&compiler_path);
                    cmd.args([
                        "-c",
                        cuda_source_path.to_str().unwrap(),
                        "-o",
                        &cuda_obj_path,
                        "-fPIC",
                        "-std=c++20",
                        "-D__HIP_PLATFORM_AMD__=1",
                        "-DUSE_ROCM=1",
                        &format!("-I{}", compute_include_path),
                        &format!("-I{}", code_dir.display()),
                        &format!("-I/usr/include"),
                        &format!("-I/usr/include/infiniband"),
                    ]);

                    // Add version-specific defines
                    if rocm_version.0 >= 7 {
                        cmd.arg("-DROCM_7_PLUS=1");
                    } else {
                        cmd.arg("-DROCM_6_X=1");
                    }

                    cmd.output()
                } else {
                    Command::new(&compiler_path)
                        .args([
                            "-c",
                            cuda_source_path.to_str().unwrap(),
                            "-o",
                            &cuda_obj_path,
                            "--compiler-options",
                            "-fPIC",
                            "-std=c++20",
                            "--expt-extended-lambda",
                            "-Xcompiler",
                            "-fPIC",
                            &format!("-I{}", compute_include_path),
                            &format!("-I{}", code_dir.display()),
                            &format!("-I/usr/include"),
                            &format!("-I/usr/include/infiniband"),
                        ])
                        .output()
                };

                match compiler_output {
                    Ok(output) => {
                        if !output.status.success() {
                            eprintln!(
                                "{} stderr: {}",
                                compiler_name,
                                String::from_utf8_lossy(&output.stderr)
                            );
                            eprintln!(
                                "{} stdout: {}",
                                compiler_name,
                                String::from_utf8_lossy(&output.stdout)
                            );
                            panic!("Failed to compile CUDA/HIP source with {}", compiler_name);
                        }
                    }
                    Err(e) => {
                        eprintln!("Failed to run {}: {}", compiler_name, e);
                        panic!("{} not found or failed to execute", compiler_name);
                    }
                }

                let ar_output = Command::new("ar")
                    .args(["rcs", &cuda_lib_path, &cuda_obj_path])
                    .output();

                match ar_output {
                    Ok(output) => {
                        if !output.status.success() {
                            eprintln!("ar stderr: {}", String::from_utf8_lossy(&output.stderr));
                            panic!("Failed to create CUDA static library with ar");
                        }
                        println!("cargo:rustc-link-lib=static=rdmaxcel_cuda");
                        println!("cargo:rustc-link-search=native={}", cuda_build_dir);
                        if let Err(e) = std::fs::copy(
                            &cuda_lib_path,
                            format!("{}/librdmaxcel_cuda.a", out_path.display()),
                        ) {
                            eprintln!("Warning: Failed to copy CUDA library to OUT_DIR: {}", e);
                        }
                    }
                    Err(e) => {
                        eprintln!("Failed to run ar: {}", e);
                        panic!("ar not found or failed to execute");
                    }
                }
            } else {
                println!(
                    "cargo:warning=CUDA/HIP source file {} not found, skipping compilation.",
                    cuda_source_path.display()
                );
            }
        }
        Err(_) => {
            println!("Note: OUT_DIR not set, skipping bindings file generation");
        }
    }
}
