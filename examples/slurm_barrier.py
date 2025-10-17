# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# @noautodeps
# pyre-ignore-all-errors
import asyncio
import logging
import os
import os
import torch
import torch.distributed as dist
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

from monarch.tools import commands
from monarch.actor import Actor, current_rank, endpoint
from monarch.actor import Actor, current_rank, endpoint
from monarch.utils import setup_env_for_distributed
from torch.nn.parallel import DistributedDataParallel as DDP
from slurm.utils import get_appdef, get_server_info, create_proc_mesh

os.environ["RUST_BACKTRACE"] = "full"
os.environ["RUST_LOG"] = "debug"
os.environ["SBATCH_RESERVATION"] = "MonarchDDP"
os.environ["TORCH_NCCL_TRACE_BUFFER_SIZE"] = "1000000"


logging.basicConfig(
    level=logging.DEBUG,
    format="%(name)s %(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)


logger: logging.Logger = logging.getLogger(__name__)

class BarrierActor(Actor):
    """This Actor wraps the basic functionality from Torch's DDP example.

    Conveniently, all of the methods we need are already laid out for us,
    so we can just wrap them in the usual Actor endpoint semantic with some
    light modifications.

    Adapted from: https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html#basic-use-case
    """

    def __init__(self):
        os.environ["NCCL_DEBUG"] = "INFO"
        self.rank = current_rank().rank

    def _rprint(self, msg):
        """Helper method to print with rank information."""
        print(f"{self.rank=} {msg}")

    @endpoint
    async def setup(self):
        """Initialize the PyTorch distributed process group."""
        self._rprint("Initializing torch distributed")

        WORLD_SIZE = int(os.environ["WORLD_SIZE"])
        LOCAL_RANK = int(os.environ["LOCAL_RANK"])
        
        # initialize the process group
        dist.init_process_group("nccl", rank=self.rank, world_size=WORLD_SIZE, device_id=LOCAL_RANK)
        self._rprint("Finished initializing torch distributed")

    @endpoint
    async def cleanup(self):
        """Clean up the PyTorch distributed process group."""
        self._rprint("Cleaning up torch distributed")
        dist.destroy_process_group()

    @endpoint
    async def demo_basic(self):
        LOCAL_RANK = int(os.environ["LOCAL_RANK"])
        """Run a basic DDP training example."""
        self._rprint(f"{os.environ['NCCL_DEBUG']=}")
        torch.cuda.set_device(LOCAL_RANK)
        self._rprint("Running basic DDP example")
        self._rprint(f"{torch.cuda.device_count()=}")
        self._rprint(f"{torch.cuda.current_device()=}")
        self._rprint(f"{torch.cuda.get_device_name(0)=}")
        self._rprint(f"{torch.cuda.is_initialized()=}")
        t = current_rank().rank * torch.ones(1).to(f'cuda:{LOCAL_RANK}')
        torch.distributed.all_reduce(t)
        self._rprint(f"{t=}")
        self._rprint("Finished running basic DDP example")


async def main():
    num_hosts = 2
    appdef = await get_appdef(num_hosts)
    server_info = await get_server_info(appdef)

    try:
        print("CREATE PROC MESH")
        proc_mesh = await create_proc_mesh(num_hosts, appdef, server_info)
        
        await proc_mesh.logging_option(
            stream_to_client=True,
            aggregate_window_sec=None,
        )

        print("SPAWN ACTORS")
        barrier_actor = proc_mesh.spawn("barrier_actor", BarrierActor)
        print("SETUP ENV")
        await setup_env_for_distributed(proc_mesh)
        print("SETUP CALL")
        await barrier_actor.setup.call()
        print("BASIC DEMO CALL")
        await barrier_actor.demo_basic.call()
        print("CLEAUP CALL")
        await barrier_actor.cleanup.call()

        print("DDP example completed successfully!")

    finally:
        commands.kill(f"slurm:///{server_info.name}")


if __name__ == "__main__":
    asyncio.run(main())
