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

from monarch._rust_bindings.monarch_hyperactor.alloc import AllocConstraints, AllocSpec
from monarch._rust_bindings.monarch_hyperactor.shape import Extent
from monarch._src.actor.allocator import RemoteAllocator, TorchXRemoteAllocInitializer
from monarch._src.actor.host_mesh import _bootstrap_cmd

from monarch.actor import ProcMesh, HostMesh
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
    # similar to Docker image; should contain a conda env in the $img_root/conda/ directory
    # when config.workspace is not None, an ephemeral fbpkg version is created
    # that conda-packs the currently active local conda env AND the directory specified by workspace
    image = "monarch_default_workspace:latest"

    appdef = hyperactor.host_mesh(
        image=image,
        meshes=[f"mesh0:{num_hosts}:{host_type}"],  # mesh_name:num_hosts:host_type
    )
    return appdef


async def get_server_info(appdef, host_memory: int = HOST_MEMORY):
    jobname = f"monarch-{USER}"

    # TODO: Register this so we don't have to do this every time
    for role in appdef.roles:
        role.resource.memMB = host_memory

    config = Config(
        scheduler="slurm",
        appdef=appdef,
        workspace=str(CWD),  # or None to disable building ephemeral,
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

    # Use v1 API with RemoteAllocator
    # Note: RemoteAllocator requires at least 2 dimensions (hosts and sub-host like GPUs)
    host_mesh = HostMesh.allocate_nonblocking(
        "hosts",
        Extent(["hosts", "gpus"], [num_hosts, num_gpus_per_host]),
        allocator,
        bootstrap_cmd=_bootstrap_cmd(),
    )

    # Wait for host_mesh to be initialized
    await host_mesh.initialized

    # Convert HostMesh to ProcMesh by calling spawn_procs with empty dict
    # The processes are already spawned by RemoteAllocator, this just wraps them
    proc_mesh = host_mesh.spawn_procs()

    return proc_mesh
