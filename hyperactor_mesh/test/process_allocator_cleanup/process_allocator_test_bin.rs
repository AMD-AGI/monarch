/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/// Test binary for ProcessAllocator child process cleanup behavior.
/// This binary creates a ProcessAllocator and spawns several child processes,
/// then keeps running until killed. It's designed to test whether child
/// processes are properly cleaned up when the parent process is killed.
use std::time::Duration;

use hyperactor_mesh::alloc::Alloc;
use hyperactor_mesh::alloc::AllocConstraints;
use hyperactor_mesh::alloc::AllocSpec;
use hyperactor_mesh::alloc::Allocator;
use hyperactor_mesh::alloc::ProcState;
use hyperactor_mesh::alloc::ProcessAllocator;
use ndslice::shape;
use tokio::process::Command;
use tokio::time::sleep;

fn emit_proc_state(state: &ProcState) {
    if let Ok(json) = serde_json::to_string(state) {
        println!("{}", json);
        // Flush immediately to ensure parent can read events in real-time
        use std::io::Write;
        use std::io::{self};
        io::stdout().flush().unwrap();
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing to stderr to avoid interfering with JSON output
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .init();

    let bootstrap_path = buck_resources::get("monarch/hyperactor_mesh/bootstrap").unwrap();
    eprintln!("Bootstrap cmd: {:?}", bootstrap_path);
    let cmd = Command::new(&bootstrap_path);
    let mut allocator = ProcessAllocator::new(cmd);

    // Create an allocation with 4 child processes
    let mut alloc = allocator
        .allocate(AllocSpec {
            shape: shape! { replica = 4 },
            constraints: AllocConstraints::default(),
        })
        .await?;

    // Wait for all children to be running
    let mut running_count = 0;
    while running_count < 4 {
        match alloc.next().await {
            Some(state) => {
                emit_proc_state(&state);

                match &state {
                    ProcState::Running { .. } => {
                        running_count += 1;
                    }
                    ProcState::Failed { description, .. } => {
                        return Err(format!("Allocation failed: {}", description).into());
                    }
                    _ => {}
                }
            }
            None => {
                break;
            }
        }
    }

    // Keep the process running indefinitely
    // In the test, we'll kill this process and check if children are cleaned up
    loop {
        #[allow(clippy::disallowed_methods)]
        sleep(Duration::from_secs(1)).await;
    }
}
