/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::env;
use std::fs;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;

// --- HIPify Helper Functions ---

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

/// Applies the required 'CUstream_st' typedef fix to the hipified header.
fn patch_hipified_header(hipified_file_path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:warning=Patching hipified header for CUstream_st typedef...");

    let hip_typedef = "\n// HIP/ROCm Fix: Manually define CUstream_st for cxx bindings\ntypedef struct ihipStream_t CUstream_st;\n";

    // Read the hipified content
    let original_content = fs::read_to_string(hipified_file_path)?;

    // Find the end of the initial includes (usually after the last #include)
    // We'll insert the typedef before the first non-include line or at the very end.
    let lines: Vec<&str> = original_content.lines().collect();
    let mut insert_index = 0;

    for (i, line) in lines.iter().enumerate() {
        if !line.trim().starts_with("#include")
            && !line.trim().is_empty()
            && !line.trim().starts_with("//")
        {
            insert_index = i;
            break;
        }
        if i == lines.len() - 1 {
            insert_index = lines.len();
        }
    }

    let mut new_content = String::new();
    for (i, line) in lines.iter().enumerate() {
        if i == insert_index {
            new_content.push_str(hip_typedef);
        }
        new_content.push_str(line);
        new_content.push('\n');
    }

    // Rewrite the file
    fs::write(
        hipified_file_path,
        new_content.trim_end_matches('\n').as_bytes(),
    )?;

    println!("cargo:warning=Successfully injected CUstream_st typedef.");
    Ok(())
}

/// Runs `hipify_torch` on the source file.
/// Returns the path to the newly hipified header file (e.g., /.../hipified_src/wrapper_hip.h).
fn hipify_source_header(
    python_interpreter: &Path,
    src_dir: &Path,     // This is the 'cuda-sys/src' directory
    hip_src_dir: &Path, // Output directory (e.g., OUT_DIR/hipified_src)
    file_name: &str,    // "wrapper.h"
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    println!(
        "cargo:warning=Copying source header {} to {} for in-place hipify...",
        file_name,
        hip_src_dir.display()
    );
    fs::create_dir_all(hip_src_dir)?;

    let src_file = src_dir.join(file_name);
    let dest_file = hip_src_dir.join(file_name);

    if src_file.exists() {
        fs::copy(&src_file, &dest_file)?;
        println!("cargo:rerun-if-changed={}", src_file.display());
    } else {
        return Err(format!("Source file {} not found", src_file.display()).into());
    }

    println!("cargo:warning=Running hipify_torch in-place on copied sources with --v2...");

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);
    // Path traversal: cuda-sys -> monarch/
    let project_root = manifest_dir
        .parent()
        .ok_or("Failed to find project root: manifest parent not found")?;

    let hipify_script = project_root
        .join("deps")
        .join("hipify_torch")
        .join("hipify_cli.py");

    if !hipify_script.exists() {
        return Err(format!("hipify_cli.py not found at {}", hipify_script.display()).into());
    }
    println!("cargo:rerun-if-changed={}", hipify_script.display());

    // Run hipify on the copied directory
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

    println!("cargo:warning=Successfully hipified {} source", file_name);

    // The hipified output file name is wrapper_hip.h inside the hip_src_dir based on your manual run.
    const HIP_OUTPUT_NAME: &str = "wrapper_hip.h";
    let hip_file = hip_src_dir.join(HIP_OUTPUT_NAME);

    // --- PATCHING STEP ---
    if hip_file.exists() {
        patch_hipified_header(&hip_file)?;
        Ok(hip_file)
    } else {
        // Fallback: If the file was modified in place (less likely with --v2)
        let fallback_file = hip_src_dir.join(file_name);
        if fallback_file.exists() {
            patch_hipified_header(&fallback_file)?;
            Ok(fallback_file)
        } else {
            Err(format!(
                "Hipified output file not found. Expected: {}",
                hip_file.display()
            )
            .into())
        }
    }
}

// --- Main Build Logic ---

#[cfg(target_os = "macos")]
fn main() {}

#[cfg(not(target_os = "macos"))]
fn main() {
    // Constants
    const CUDA_HEADER_NAME: &str = "wrapper.h";

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Check if we are building for ROCm (HIP)
    let is_rocm = build_utils::find_rocm_home().is_some();

    println!("cargo:rerun-if-env-changed=USE_ROCM");

    let header_path;
    let compute_lib_names;
    let compute_config;

    // --- 1. HIPify or Select CUDA Header ---
    if is_rocm {
        println!("cargo::warning=Using HIP from ROCm installation");
        compute_lib_names = vec!["amdhip64"];

        // HIPify the CUDA wrapper header
        let hip_src_dir = out_dir.join("hipified_src");
        let python_interpreter = find_python_interpreter();

        header_path = hipify_source_header(
            &python_interpreter,
            &manifest_dir.join("src"), // Source is cuda-sys/src
            &hip_src_dir,
            CUDA_HEADER_NAME,
        )
        .expect("Failed to hipify wrapper.h");

        // Discover ROCm configuration
        match build_utils::discover_rocm_config() {
            Ok(config) => {
                compute_config = build_utils::CudaConfig {
                    cuda_home: config.rocm_home,
                    include_dirs: config.include_dirs,
                    lib_dirs: config.lib_dirs,
                }
            }
            Err(_) => {
                build_utils::print_rocm_error_help();
                std::process::exit(1);
            }
        }
    } else {
        println!("cargo::warning=Using CUDA");
        compute_lib_names = vec!["cuda", "cudart"];
        header_path = manifest_dir.join("src").join(CUDA_HEADER_NAME);

        // Discover CUDA configuration
        match build_utils::discover_cuda_config() {
            Ok(config) => compute_config = config,
            Err(_) => {
                build_utils::print_cuda_error_help();
                std::process::exit(1);
            }
        }
    }

    // --- 2. Configure bindgen ---
    let mut builder = bindgen::Builder::default()
        // The input header we would like to generate bindings for
        .header(header_path.to_str().expect("Invalid header path"))
        .clang_arg("-x")
        .clang_arg("c++")
        .clang_arg("-std=gnu++20")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        // Allow the specified functions and types (now generated from the appropriate header)
        .allowlist_function("cu.*")
        .allowlist_function("CU.*")
        .allowlist_type("cu.*")
        .allowlist_type("CU.*")
        // Conditionally allow HIP functions and types for ROCm (HIPify changes cuda to hip)
        .allowlist_function("hip.*")
        .allowlist_type("hip.*")
        // Use newtype enum style
        .default_enum_style(bindgen::EnumVariation::NewType {
            is_bitfield: false,
            is_global: false,
        });

    // Add CUDA/ROCm include paths from the discovered configuration
    for include_dir in &compute_config.include_dirs {
        builder = builder.clang_arg(format!("-I{}", include_dir.display()));
    }

    // Define HIP platform macros when using ROCm (needed for bindgen preprocessor logic)
    if is_rocm {
        builder = builder
            .clang_arg("-D__HIP_PLATFORM_AMD__=1")
            .clang_arg("-DUSE_ROCM=1");
    }

    // --- 3. Link Python environment ---
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
        println!("cargo::metadata=LIB_PATH={}", lib_dir);
    }

    // --- 4. Link Compute Libraries ---
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

    // --- 5. Generate Bindings ---
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
