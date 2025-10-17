# %%
#%cd /mnt/models/mreso/monarch/examples/

# %%
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

# %% [markdown]
# ## Monarch + TorchTitan on SLURM
# This example notebook demonstrates how you can easily run and iterate on a distributed training job with Monarch and TorchTitan.
# 
# #### Prerequisites
# Please make sure your environment is setup for this notebook:
# 1. Install Monarch nightly: https://github.com/meta-pytorch/monarch/blob/main/scripts/install_nightly.py
# 2. Install Titan nightly: https://github.com/pytorch/torchtitan?tab=readme-ov-file#nightly-builds
# 3. Ensure you have a valid Titan model config in the script directory (i.e: https://github.com/pytorch/torchtitan/blob/main/torchtitan/models/llama3/train_configs/debug_model.toml)

# %% [markdown]
# ### 1. Reserve your SLURM job
# If necessary, update paramaters for your cluster:
# - host_type: TorchX named resource for your cluster (default: "gpu.xlarge")
# - host_memory: Memory per machine in MB (default: 2062607)
# 
# For more information on TorchX resources: https://docs.pytorch.org/torchx/main/specs.html#resource

# %%
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
from slurm.utils_with_init import get_appdef, get_server_info, create_proc_mesh
num_nodes = 2 # assign for your system

async def setup():


    appdef = await get_appdef(
        num_nodes,
        # host_type = ...
    )
    server_info = await get_server_info(
        appdef,
        # host_memory = ...
    )

    return appdef, server_info

# %% [markdown]
# ### 2. Define your Titan and cluster parameters

# %%
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from torchtitan.train import Trainer
from torchtitan.config import ConfigManager, JobConfig
from monarch.actor import Actor, current_rank, endpoint
from torchtitan.tools.logging import init_logger, logger
import torch
from dataclasses import dataclass
import os
from monarch.tools import commands
from monarch.utils import setup_env_for_distributed


@dataclass
class RunParams:
    """
        Parameters for your cluster and training job, adjust as needed
    """
    training_steps: int = 50
    model_config = "/mnt/models/mreso/torchtitan/torchtitan/models/llama3/train_configs/debug_model.toml"
    # model_config = "/mnt/models/mreso/torchtitan/torchtitan/models/llama3/train_configs/llama3_8b.toml"
    # model_config = "/mnt/models/mreso/torchtitan/torchtitan/models/llama3/train_configs/llama3_70b.toml"
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
        "--model.tokenizer_path",
        "/mnt/models/mreso/torchtitan/tests/assets/tokenizer/",
        # "/mnt/models/mreso/torchtitan/assets/hf/Llama-3.1-8B/",
        # "/mnt/models/mreso/torchtitan/assets/hf/Llama-3.1-70B/",
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

# %% [markdown]
# ### 3. Execute your training job
# You can make adjustments and run this on the existing SLURM allocations as many times as you would like!

# %%
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

async def main():
    appdef, server_info = await setup()

    job_config = make_job_config()
    proc_mesh = None

    try:
        # 1. Create a proc mesh on your SLURM allocation
        print("CREATING PROC MESH")
        proc_mesh = await create_proc_mesh(RunParams.num_nodes, appdef, server_info)
        
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
        trainer = await proc_mesh.spawn("trainer_actor", TrainerActor, job_config)
        # 4. Execute the taining job
        print("CALLING TRAINER")
        await trainer.start_training.call()
    except Exception as e:
        logger.info(f"Trainer failed: {e}")
    finally:
        if proc_mesh:
            await proc_mesh.stop()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

# %% [markdown]
# ### 4. Destory the SLURM job
# Once you're done experimenting, free up the allocation

# %%
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# commands.kill(f"slurm:///{server_info.name}")


