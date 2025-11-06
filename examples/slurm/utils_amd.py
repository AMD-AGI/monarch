# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import getpass
import json
import logging
import os
import pathlib
import subprocess

from monarch._rust_bindings.monarch_hyperactor.alloc import AllocConstraints, AllocSpec
from monarch._src.actor.allocator import RemoteAllocator, TorchXRemoteAllocInitializer

from monarch.actor import ProcMesh
from monarch.tools import commands
from monarch.tools.components import hyperactor
from monarch.tools.config import Config


USER = getpass.getuser()
HOME = pathlib.Path().home()
CWD = os.getcwd()
DEACTIVATE = None

logging.basicConfig(
    level=logging.INFO,
    format="%(name)s %(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)
logger: logging.Logger = logging.getLogger(__name__)

# pre-configured for H100
HOST_TYPE = "gpu.xlarge"
HOST_MEMORY = 2062607


async def get_appdef(num_hosts: int, host_type: str = HOST_TYPE):
    """
    Create application definition using the docker exec bootstrap script.
    This assumes containers are initialized separately.
    """
    image = "monarch_default_workspace:latest"

    appdef = hyperactor.host_mesh(
        image=image,
        meshes=[f"mesh0:{num_hosts}:{host_type}"],
        # Use the exec-only bootstrap script
        program='/home/mreso/monarch/examples/custom_bootstrap_exec.sh',
    )
    return appdef


async def get_server_info(appdef, host_memory: int = HOST_MEMORY, exclude_nodes: str = "chi2599,chi2600,chi2602,chi2603"):
    """
    Get or create server info for SLURM allocation.
    
    Args:
        appdef: Application definition
        host_memory: Memory per host in MB
        exclude_nodes: Comma-separated list of nodes to exclude (default: "chi2599,chi2600")
                      Set to None or empty string to disable exclusion.
    """
    jobname = f"monarch-{USER}"

    for role in appdef.roles:
        role.resource.memMB = host_memory

    # Pass exclude parameter directly to SLURM scheduler
    scheduler_args = {}
    if exclude_nodes:
        scheduler_args["exclude"] = exclude_nodes
        logger.info(f"Excluding SLURM nodes: {exclude_nodes}")

    config = Config(
        scheduler="slurm",
        appdef=appdef,
        workspace=str(CWD),
        scheduler_args=scheduler_args,
    )

    server_info = await commands.get_or_create(
        jobname,
        config,
        force_restart=False,
    )
    
    return server_info


async def create_proc_mesh(num_hosts, appdef, server_info):
    num_gpus_per_host = appdef.roles[0].resource.gpu

    logger.info(
        "\n===== Server Info =====\n%s",
        json.dumps(server_info.to_json(), indent=2),
    )

    allocator = RemoteAllocator(
        world_id="foo",
        initializer=TorchXRemoteAllocInitializer(server_info.server_handle),
    )
    alloc = allocator.allocate(
        AllocSpec(AllocConstraints(), hosts=num_hosts, gpus=num_gpus_per_host)
    )

    proc_mesh = ProcMesh.from_alloc(alloc)
    return proc_mesh


def init_docker_containers(num_hosts: int, server_info=None):
    """
    Initialize Docker containers on each SLURM node BEFORE creating the process mesh.
    Uses srun to execute init script on allocated nodes.
    
    Args:
        num_hosts: Number of hosts/nodes to initialize containers on
        server_info: Server info from get_server_info() (required when called from client)
    """
    logger.info(f"Initializing Docker containers on {num_hosts} SLURM node(s)")
    
    init_script = '/mnt/models/mreso/monarch/examples/init_docker_container.sh'
    
    # Make sure the script is executable
    os.chmod(init_script, 0o755)
    
    try:
        # Check if we're in a SLURM environment
        job_id = os.environ.get('SLURM_JOB_ID')
        
        if job_id:
            # We're inside a SLURM job, run directly
            logger.info(f"Running inside SLURM job {job_id}, executing init script directly")
            cmd = [init_script]
        else:
            # We're on the client, use srun to execute on allocated nodes
            logger.info("Running from client, using srun to execute on SLURM nodes")
            
            if not server_info:
                raise ValueError("server_info is required when calling init_docker_containers from client")
            
            # Get job ID from server_info
            job_id = server_info.name  # This is the SLURM job ID like "1542"
            logger.info(f"Using SLURM job ID: {job_id}")
            
            # Use srun with --jobid to run on the specific job's allocated nodes
            cmd = [
                "srun",
                f"--jobid={job_id}",
                f"--ntasks={num_hosts}",
                "--ntasks-per-node=1",
                init_script
            ]
        
        logger.info(f"Executing: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        logger.info(f"Container initialization output:\n{result.stdout}")
        if result.stderr:
            logger.warning(f"Container initialization stderr:\n{result.stderr}")
            
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to initialize containers: {e}")
        logger.error(f"stdout: {e.stdout}")
        logger.error(f"stderr: {e.stderr}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during container initialization: {e}")
        raise
    
    logger.info("Docker containers initialized successfully on SLURM nodes")


def cleanup_docker_containers(num_hosts: int, server_info=None):
    """
    Cleanup Docker containers after the job completes.
    
    Args:
        num_hosts: Number of hosts to cleanup containers on
        server_info: Server info (optional, for job ID when called from client)
    """
    logger.info(f"Cleaning up Docker containers on {num_hosts} host(s)")
    
    try:
        # Check if we're in a SLURM environment
        job_id = os.environ.get('SLURM_JOB_ID')
        
        if job_id:
            # We're inside a SLURM job, run directly
            logger.info(f"Running inside SLURM job {job_id}, executing cleanup directly")
            hostname = socket.gethostname()
            container_name = f"monarch_node_{hostname}"
            cmd = ["docker", "rm", "-f", container_name]
        else:
            # We're on the client, use srun to execute on allocated nodes
            logger.info("Running from client, using srun to execute cleanup on SLURM nodes")
            
            if server_info:
                job_id = server_info.name
                logger.info(f"Using SLURM job ID: {job_id}")
                # Use a shell command to compute container name based on hostname on each node
                cmd = [
                    "srun",
                    f"--jobid={job_id}",
                    f"--ntasks={num_hosts}",
                    "--ntasks-per-node=1",
                    "/bin/bash", "-c",
                    "docker rm -f monarch_node_$(hostname)"
                ]
            else:
                # Fallback: try to cleanup on current host only
                logger.warning("No server_info provided, attempting cleanup on current host only")
                hostname = socket.gethostname()
                container_name = f"monarch_node_{hostname}"
                cmd = ["docker", "rm", "-f", container_name]
        
        logger.info(f"Executing cleanup: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False  # Don't fail if container doesn't exist
        )
        
        if result.returncode == 0:
            logger.info(f"Container cleanup successful")
            if result.stdout:
                logger.info(f"Cleanup output:\n{result.stdout}")
        else:
            logger.warning(f"Container cleanup had issues (exit code {result.returncode})")
            if result.stderr:
                logger.warning(f"Cleanup stderr:\n{result.stderr}")
                
    except Exception as e:
        logger.error(f"Unexpected error during container cleanup: {e}")
    
    logger.info("Docker containers cleanup completed")

