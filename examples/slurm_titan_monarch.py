import os
os.environ.setdefault("GPU_MAX_HW_QUEUES", "2")
os.environ.setdefault("TORCH_NCCL_HIGH_PRIORITY", "1")
os.environ.setdefault("NCCL_CHECKS_DISABLE", "1")
os.environ.setdefault("NCCL_IB_GID_INDEX", "3")
os.environ.setdefault("NCCL_CROSS_NIC", "0")
os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
os.environ.setdefault("NCCL_PROTO", "Simple")
os.environ.setdefault("RCCL_MSCCL_ENABLE", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HSA_NO_SCRATCH_RECLAIM", "1")
os.environ.setdefault("NCCL_PXN_DISABLE", "0")
os.environ.setdefault("NCCL_P2P_NET_CHUNKSIZE", "262144")
# Enable NCCL debug logging to verify RDMA usage
os.environ.setdefault("NCCL_DEBUG", "INFO")
os.environ.setdefault("NCCL_DEBUG_SUBSYS", "INIT,NET")
# NCCL network interface configuration for multi-host communication
# These should be set externally based on your specific system configuration:
# - NCCL_IB_HCA: InfiniBand/RoCE adapters to use (e.g., "bnxt_re0,bnxt_re1,...")
# - NCCL_SOCKET_IFNAME: Network interface name for socket communication
# Fix XDG_RUNTIME_DIR to use /tmp instead of /run/user/<uid> for SLURM jobs
# SLURM processes may not have access to /run/user/<uid> even if the directory exists
os.environ["XDG_RUNTIME_DIR"] = "/tmp"

# Workaround for PyTorch 2.8.0+rocm6.4 inductor import error with scaled_mm_configs
# Disable torch.compile to avoid importing problematic modules
import torch
torch._dynamo.config.disable = True

from torchtitan.train import Trainer
from torchtitan.config import ConfigManager, JobConfig
from monarch.actor import Actor, current_rank, endpoint
from torchtitan.tools.logging import init_logger, logger
from dataclasses import dataclass
import os
from monarch.tools import commands
from monarch.utils import setup_env_for_distributed

from slurm.utils import get_appdef, get_server_info, create_proc_mesh

num_nodes = 2 # assign for your system
MONARCH_EXAMPLE_FOLDER=os.getcwd()
os.environ["MONARCH_EXAMPLE_FOLDER"]=MONARCH_EXAMPLE_FOLDER


async def setup():


    appdef = await get_appdef(
        num_nodes,
        # host_type = ...
        # Note: The bootstrap script is configured via environment variables, not passed here
        # program=f'{MONARCH_EXAMPLE_FOLDER}/custom_bootstrap_exec.sh',
    )
    server_info = await get_server_info(
        appdef,
        # host_memory = ...
    )

    return appdef, server_info



@dataclass
class RunParams:
    """
        Parameters for your cluster and training job, adjust as needed
    """
    training_steps: int = 50
    model_config = f"{MONARCH_EXAMPLE_FOLDER}/../../torchtitan/torchtitan/models/llama3/train_configs/debug_model.toml"
    dataset = "c4"
    num_nodes = num_nodes
    gpus_per_node = 8


class TrainerActor(Actor):
    """
        A simple wrapper class with executes a TorchTitan trainer in a Monarch actor
    """
    def __init__(self, job_config: JobConfig) -> None:
        self.job_config = job_config
        rank = current_rank().rank
        self.uid = f"[trainer_{rank}]"

    @endpoint
    async def start_training(self) -> None:
        init_logger()
        trainer: Trainer | None = None

        try:
            trainer = Trainer(self.job_config)
            logger.info(f"{self.uid} initialized successfully and starting training")
            trainer.train()
        except Exception:
            if trainer:
                trainer.close()
            raise
        else:
            trainer.close()
        finally:
            torch.distributed.destroy_process_group()
            logger.info(f"{self.uid} trainer cleaned up")

def make_job_config() -> JobConfig:
    """
        Create a job config which is digested by TorchTitan, sourced from RunParams
    """
    data_parallel_shard_degree = RunParams.num_nodes * RunParams.gpus_per_node
    output_path = "./outputs"

    default_args = [
        "--job.config_file",
        RunParams.model_config,
        "--model.hf_assets_path",
        f"{MONARCH_EXAMPLE_FOLDER}/../../torchtitan/tests/assets/tokenizer/",
        # f"{MONARCH_EXAMPLE_FOLDER}/../../torchtitan/assets/hf/Llama-3.1-8B/",
        "--comm.trace_buf_size",
        "0",
        "--metrics.log_freq",
        "1",
        "--parallelism.data_parallel_shard_degree",
        str(-1),
        "--activation_checkpoint.mode",
        "full",
        "--comm.train_timeout_seconds",
        "60",
        "--training.steps",
        str(RunParams.training_steps),
        "--training.dataset",
        RunParams.dataset,
        "--job.dump_folder",
        output_path,
        "--metrics.enable_tensorboard",
    ]

    config_manager = ConfigManager()
    job_config = config_manager.parse_args(default_args)

    return job_config

async def main():
    # Configure transport to TCP before any other Monarch API calls
    # RDMA will be used automatically by Monarch when available on the underlying network
    from monarch.actor import enable_transport, ChannelTransport
    enable_transport(ChannelTransport.TcpWithHostname)

    appdef, server_info = await setup()

    job_config = make_job_config()
    proc_mesh = None

    try:
        # 1. Create a proc mesh on your SLURM allocation
        print("CREATING PROC MESH")
        proc_mesh = await create_proc_mesh(RunParams.num_nodes, appdef, server_info)

        # Note: proc_mesh.initialized is already awaited in create_proc_mesh
        print("PROC MESH INITIALIZED")

        # 2. Define remote logging behavior
        await proc_mesh.logging_option(
            stream_to_client=True,
            aggregate_window_sec=None
        )
        # 3. Prepare trainer for torch distributed
        print("SETUP ENV FOR DISTRIBUTED")
        await setup_env_for_distributed(
            proc_mesh,
            )

        print("SPAWNING TRAINER")
        trainer = proc_mesh.spawn("trainer_actor", TrainerActor, job_config)
        # 4. Execute the taining job
        print("CALLING TRAINER")
        await trainer.start_training.call()
    except Exception as e:
        import traceback
        logger.info(f"Trainer failed: {e}")
        traceback.print_exc()
    finally:
        if proc_mesh:
            await proc_mesh.stop()

        commands.kill(f"slurm:///{server_info.name}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())