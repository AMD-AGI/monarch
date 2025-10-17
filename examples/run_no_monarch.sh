#!/bin/bash
#SBATCH --ntasks=2
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --nodelist=chi2644,chi2645

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
printf "Nodes allocated to this job: %s\n" "${nodes[@]}"
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

head_node_ip=$(echo $head_node_ip | cut -d " " -f 1)

export NCCL_DEBUG=INFO

# To accomodate network issues in Vultr
#We can increase the NCCL_IB_RETRY_CNT to 10 from 7, and increase NCCL_IB_TIMEOUT from 18 to 30
export NCCL_IB_RETRY_CNT=10
export NCCL_IB_TIMEOUT=30

# Exclude bnxt_re1 due to cross-node IB errors
export NCCL_IB_HCA=${NCCL_IB_HCA:="bnxt_re0,bnxt_re1,bnxt_re2,bnxt_re3,bnxt_re4,bnxt_re5,bnxt_re7,bnxt_re8"}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:="enp49s0f0np0"}

# Setup the IB mount options
if [ -e "/etc/libibverbs.d/bnxt_re.driver" ]; then
  echo "/etc/libibverbs.d exists and using broadcom."
  export IB_MOUNT_OPTIONS="-v /usr/bin:/usr/bin -v /etc/libibverbs.d/:/etc/libibverbs.d -v /usr/lib/x86_64-linux-gnu/:/usr/lib/x86_64-linux-gnu/ -v /usr/local/lib:/usr/local/lib"
else
  echo "/etc/libibverbs.d does not exist not using."
  export IB_MOUNT_OPTIONS=""
fi


# export DOCKER_IMAGE=${DOCKER_IMAGE:-"docker.io/rocm/megatron-lm:v25.6_py312"}
export DOCKER_IMAGE=${DOCKER_IMAGE:-"docker.io/library/monarch_amd"}
srun docker run --rm \
 --env NCCL_IB_HCA=${NCCL_IB_HCA} \
 --env NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME} \
 --env NCCL_DEBUG=${NCCL_DEBUG} \
 --env SLURM_NNODES=${SLURM_NNODES} \
 --env SLURM_GPUS_PER_NODE=${SLURM_GPUS_PER_NODE} \
 --env head_node_ip=${head_node_ip} \
 --ipc=host --network=host --device=/dev/kfd --device=/dev/dri \
 --cap-add=SYS_PTRACE --cap-add=CAP_SYS_ADMIN \
 --security-opt seccomp=unconfined --group-add video --privileged \
 --device=/dev/infiniband \
 -v /mnt/models/mreso/:/mnt/models/mreso/ \
 -v /mnt/models/john/:/mnt/models/john/ \
 -w /mnt/models/mreso/monarch/examples/ \
 ${DOCKER_IMAGE} \
 /bin/bash -c 'torchrun --nnodes ${SLURM_NNODES} --nproc_per_node ${SLURM_GPUS_PER_NODE} --rdzv_id 101 --rdzv_backend c10d --rdzv_endpoint "$head_node_ip:29500" slurm_barrier_mreso_no_monarch.py'

# #  ${IB_MOUNT_OPTIONS} \

# srun torchrun --nnodes ${SLURM_NNODES} --nproc_per_node ${SLURM_GPUS_PER_NODE} --rdzv_id 101 --rdzv_backend c10d --rdzv_endpoint "$head_node_ip:29500" slurm_barrier_mreso_no_monarch.py