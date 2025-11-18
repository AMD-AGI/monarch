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

/// Runs `hipify_torch` on the source directory.
fn hipify_sources(
    python_interpreter: &Path,
    src_dir: &Path,
    hip_src_dir: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "cargo:warning=Copying sources from {} to {} for in-place hipify...",
        src_dir.display(),
        hip_src_dir.display()
    );
    fs::create_dir_all(hip_src_dir)?;

    let files_to_copy = ["cuda_ping_pong.cu", "cuda_ping_pong.cuh"];

    for file_name in files_to_copy {
        let src_file = src_dir.join(file_name);
        let dest_file = hip_src_dir.join(file_name);
        if src_file.exists() {
            fs::copy(&src_file, &dest_file)?;
            println!("cargo:rerun-if-changed={}", src_file.display());
        } else {
            return Err(format!("Source file {} not found", src_file.display()).into());
        }
    }

    println!("cargo:warning=Running hipify_torch in-place on copied sources with --v2...");

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);
    let project_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .and_then(|p| p.parent())
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

    println!("cargo:warning=Successfully hipified cuda_ping_pong sources");

    // Debug: List what files were actually created
    if let Ok(entries) = fs::read_dir(hip_src_dir) {
        println!("cargo:warning=Files in hipified directory:");
        for entry in entries.filter_map(|e| e.ok()) {
            println!("cargo:warning=  - {}", entry.file_name().to_string_lossy());
        }
    }

    Ok(())
}

/// Post-process hipified files to fix include paths
fn patch_hipified_files(hip_src_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:warning=Patching hipified cuda_ping_pong files...");

    // Patch the .hip file (hipified .cu) - hipify_torch --v2 creates hip_ping_pong.hip
    let hip_file = hip_src_dir.join("hip_ping_pong.hip");
    if hip_file.exists() {
        let content = fs::read_to_string(&hip_file)?;

        let patched_content = content
            // Fix the include path for rdmaxcel - it should use rdmaxcel_hip.h
            .replace(
                "#include <monarch/rdmaxcel-sys/src/rdmaxcel.h>",
                "#include \"rdmaxcel_hip.h\"",
            )
            .replace(
                "#include \"monarch/rdmaxcel-sys/src/rdmaxcel.h\"",
                "#include \"rdmaxcel_hip.h\"",
            );

        fs::write(&hip_file, patched_content)?;
        println!("cargo:warning=Patched {}", hip_file.display());
    }

    // Patch the .cuh file (header) - hipify_torch --v2 creates hip_ping_pong.cuh
    let cuh_file = hip_src_dir.join("hip_ping_pong.cuh");
    if cuh_file.exists() {
        let content = fs::read_to_string(&cuh_file)?;

        let patched_content = content
            // Fix the include path for rdmaxcel
            .replace(
                "#include \"monarch/rdmaxcel-sys/src/rdmaxcel.h\"",
                "#include \"rdmaxcel_hip.h\"",
            )
            .replace(
                "#include <monarch/rdmaxcel-sys/src/rdmaxcel.h>",
                "#include \"rdmaxcel_hip.h\"",
            );

        fs::write(&cuh_file, patched_content)?;
        println!("cargo:warning=Patched {}", cuh_file.display());
    }

    println!("cargo:warning=Applied post-processing fixes to hipified cuda_ping_pong files");
    Ok(())
}

fn main() {
    // Determine if we're building for ROCm
    let is_rocm = env::var("USE_ROCM").is_ok();

    println!("cargo:rerun-if-changed=cuda_ping_pong.cu");
    println!("cargo:rerun-if-changed=cuda_ping_pong.cuh");
    println!("cargo:rerun-if-env-changed=USE_ROCM");

    if is_rocm {
        build_hip();
    } else {
        build_cuda();
    }
}

fn build_cuda() {
    println!("cargo:warning=Building cuda_ping_pong with CUDA");

    // Find rdmaxcel-sys source directory for includes
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let rdmaxcel_src = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .and_then(|p| p.parent())
        .map(|p| p.join("rdmaxcel-sys").join("src"))
        .expect("Could not find rdmaxcel-sys/src");

    // Compile with nvcc
    cc::Build::new()
        .cuda(true)
        .flag("-cudart=shared")
        .flag("-gencode")
        .flag("arch=compute_80,code=sm_80") // Adjust for your GPU
        .include(&rdmaxcel_src)
        .file("cuda_ping_pong.cu")
        .compile("cuda_ping_pong");

    println!("cargo:rustc-link-lib=cudart");

    // Search for CUDA libs
    if let Ok(cuda_path) = env::var("CUDA_PATH") {
        println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
    } else {
        println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    }
}

fn build_hip() {
    println!("cargo:warning=Building cuda_ping_pong with HIP/ROCm");

    // Get ROCm home
    let rocm_home = env::var("ROCM_PATH")
        .or_else(|_| env::var("ROCM_HOME"))
        .unwrap_or_else(|_| "/opt/rocm".to_string());

    println!("cargo:warning=Using ROCm from: {}", rocm_home);

    // Detect ROCm version
    let rocm_version = get_rocm_version(&rocm_home).unwrap_or((6, 0));
    println!(
        "cargo:warning=Using ROCm version {}.{}",
        rocm_version.0, rocm_version.1
    );

    // Get current directory and output directory
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Create hipified source directory
    let hip_src_dir = out_dir.join("hipified_src");

    // Find Python interpreter
    let python_interpreter = find_python_interpreter();

    // Hipify the sources
    hipify_sources(&python_interpreter, &manifest_dir, &hip_src_dir)
        .expect("Failed to hipify cuda_ping_pong sources");

    // Find the hipified rdmaxcel-sys directory
    // Navigate to the build directory and search for rdmaxcel-sys build output
    let build_dir = out_dir
        .ancestors()
        .find(|p| p.ends_with("build"))
        .expect("Could not find build directory");

    // Look for ALL directories starting with "rdmaxcel-sys-" and find the most recent one
    // that actually has the hipified_src directory with rdmaxcel_hip.h
    let rdmaxcel_hipified = std::fs::read_dir(build_dir)
        .expect("Failed to read build directory")
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.file_name()
                .to_string_lossy()
                .starts_with("rdmaxcel-sys-")
        })
        .filter_map(|e| {
            let hipified_path = e.path().join("out").join("hipified_src");
            let header_path = hipified_path.join("rdmaxcel_hip.h");
            // Only consider this directory if it has the hipified header
            if header_path.exists() {
                // Get the modification time to find the most recent one
                if let Ok(metadata) = fs::metadata(&header_path) {
                    if let Ok(modified) = metadata.modified() {
                        return Some((hipified_path, modified));
                    }
                }
            }
            None
        })
        .max_by_key(|(_, modified)| *modified)
        .map(|(path, _)| path)
        .expect("Could not find rdmaxcel-sys hipified sources with rdmaxcel_hip.h. Make sure rdmaxcel-sys is built with USE_ROCM=1 first!");

    println!(
        "cargo:warning=Using rdmaxcel headers from: {}",
        rdmaxcel_hipified.display()
    );

    // Patch the hipified files to fix include paths
    patch_hipified_files(&hip_src_dir).expect("Failed to patch hipified files");

    // Now compile with hipcc - try to find the hipified .cu file
    // hipify_torch with --v2 creates files with "hip_" prefix: hip_ping_pong.hip
    let possible_hip_files = [
        hip_src_dir.join("hip_ping_pong.hip"),  // Most likely with --v2
        hip_src_dir.join("cuda_ping_pong.hip"), // Alternative naming
        hip_src_dir.join("cuda_ping_pong_hip.cu"), // Another alternative
    ];

    let hip_file = possible_hip_files
        .iter()
        .find(|f| f.exists())
        .cloned()
        .unwrap_or_else(|| {
            // List what files actually exist
            println!("cargo:warning=Looking for hipified source file, but not found. Files in directory:");
            if let Ok(entries) = fs::read_dir(&hip_src_dir) {
                for entry in entries.filter_map(|e| e.ok()) {
                    println!("cargo:warning=  - {}", entry.path().display());
                }
            }
            panic!(
                "Hipified source file not found. Tried: {:?}",
                possible_hip_files.iter().map(|p| p.display().to_string()).collect::<Vec<_>>()
            );
        });

    println!(
        "cargo:warning=Using hipified source file: {}",
        hip_file.display()
    );

    // Find rdmaxcel.hip to compile together with our code (device functions need to be compiled together)
    let rdmaxcel_hip_source = rdmaxcel_hipified.join("rdmaxcel.hip");
    if !rdmaxcel_hip_source.exists() {
        panic!(
            "rdmaxcel.hip not found at {}. Device functions need to be compiled together.",
            rdmaxcel_hip_source.display()
        );
    }
    println!(
        "cargo:warning=Found rdmaxcel.hip: {}",
        rdmaxcel_hip_source.display()
    );

    // For HIP device code, we need to compile with relocatable device code (RDC)
    // This allows device symbols to be resolved at link time
    let hipcc_path = format!("{}/bin/hipcc", rocm_home);
    let ping_pong_obj = out_dir.join("hip_ping_pong.o");
    let rdmaxcel_obj = out_dir.join("rdmaxcel.o");

    let common_args = vec![
        format!("-I{}/include", rocm_home),
        format!("-I{}", hip_src_dir.display()),
        format!("-I{}", rdmaxcel_hipified.display()),
        "-D__HIP_PLATFORM_AMD__".to_string(),
        "-DUSE_ROCM".to_string(),
        "-std=c++17".to_string(),
        "-fPIC".to_string(),
        "--offload-arch=gfx90a".to_string(),
        "-fgpu-rdc".to_string(), // Enable relocatable device code
    ];

    let mut version_flag = Vec::new();
    if rocm_version.0 >= 7 {
        version_flag.push("-DROCM_7_PLUS".to_string());
    } else if rocm_version.0 >= 6 {
        version_flag.push("-DROCM_6_X".to_string());
    }

    // Compile hip_ping_pong.hip with RDC
    let mut compile_ping_pong = Command::new(&hipcc_path);
    compile_ping_pong
        .arg("-c")
        .arg(&hip_file)
        .arg("-o")
        .arg(&ping_pong_obj)
        .args(&common_args)
        .args(&version_flag);

    println!("cargo:warning=Compiling hip_ping_pong.hip with RDC...");
    let output = compile_ping_pong.output().expect("Failed to run hipcc");
    if !output.status.success() {
        eprintln!("hipcc stdout: {}", String::from_utf8_lossy(&output.stdout));
        eprintln!("hipcc stderr: {}", String::from_utf8_lossy(&output.stderr));
        panic!("hipcc failed to compile hip_ping_pong.hip");
    }

    // Compile rdmaxcel.hip with RDC
    let mut compile_rdmaxcel = Command::new(&hipcc_path);
    compile_rdmaxcel
        .arg("-c")
        .arg(&rdmaxcel_hip_source)
        .arg("-o")
        .arg(&rdmaxcel_obj)
        .args(&common_args)
        .args(&version_flag);

    println!("cargo:warning=Compiling rdmaxcel.hip with RDC...");
    let output = compile_rdmaxcel.output().expect("Failed to run hipcc");
    if !output.status.success() {
        eprintln!("hipcc stdout: {}", String::from_utf8_lossy(&output.stdout));
        eprintln!("hipcc stderr: {}", String::from_utf8_lossy(&output.stderr));
        panic!("hipcc failed to compile rdmaxcel.hip");
    }

    // Link the object files together with device linking
    let linked_obj = out_dir.join("cuda_ping_pong_linked.o");
    let mut link_cmd = Command::new(&hipcc_path);
    link_cmd
        .arg("-fgpu-rdc")
        .arg("--offload-arch=gfx90a")
        .arg("-r") // Partial link - create relocatable object, not executable
        .arg("-o")
        .arg(&linked_obj)
        .arg(&ping_pong_obj)
        .arg(&rdmaxcel_obj);

    println!("cargo:warning=Linking object files with device linker...");
    let output = link_cmd.output().expect("Failed to run hipcc for linking");
    if !output.status.success() {
        eprintln!(
            "hipcc link stdout: {}",
            String::from_utf8_lossy(&output.stdout)
        );
        eprintln!(
            "hipcc link stderr: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        panic!("hipcc failed to link object files");
    }

    // Create a static library from the linked object file
    let lib_file = out_dir.join("libcuda_ping_pong_hip.a");
    let ar_output = Command::new("ar")
        .args(&[
            "rcs",
            lib_file.to_str().unwrap(),
            linked_obj.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to run ar");

    if !ar_output.status.success() {
        eprintln!("ar stderr: {}", String::from_utf8_lossy(&ar_output.stderr));
        panic!("ar failed to create static library");
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=cuda_ping_pong_hip");

    // Link against HIP runtime
    println!("cargo:rustc-link-search=native={}/lib", rocm_home);
    println!("cargo:rustc-link-lib=amdhip64");
}
