#!/bin/bash

HOST_MOUNT="/mnt/models/mreso/monarch"     # change this path to host dir intend to be attached to the docker
CONTAINER_MOUNT=${HOST_MOUNT}      # change this path to development workspace path inside the docker

# Define the Docker image
export NCCL_IB_HCA=${NCCL_IB_HCA:="bnxt_re0,bnxt_re1,bnxt_re2,bnxt_re3,bnxt_re4,bnxt_re5,bnxt_re7,bnxt_re8"} # modify based on the GPU NiC settings
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:="enp49s0f0np0"}

# Setup the IB mount options
if [ -e "/etc/libibverbs.d/bnxt_re.driver" ]; then
  echo "/etc/libibverbs.d exists and using broadcom."
  export IB_MOUNT_OPTIONS="-v /usr/bin:/usr/bin -v /etc/libibverbs.d/:/etc/libibverbs.d -v /usr/lib/x86_64-linux-gnu/:/usr/lib/x86_64-linux-gnu/ -v /usr/local/lib:/usr/local/lib"
  
else
  echo "/etc/libibverbs.d does not exist not using ."
  export IB_MOUNT_OPTIONS=""
fi

export DOCKER_IMAGE=${DOCKER_IMAGE:-"docker.io/rocm/megatron-lm:v25.5_py310"}

bash -c "docker ps -aq | xargs -r docker rm -f ; \
docker run --rm -ti \
 --env NCCL_IB_HCA=\$NCCL_IB_HCA \
 --env NCCL_SOCKET_IFNAME=\$NCCL_SOCKET_IFNAME \
 --env CONTAINER_MOUNT=\$CONTAINER_MOUNT \
 --env HYPERACTOR_MESH_BOOTSTRAP_ADDR=\$HYPERACTOR_MESH_BOOTSTRAP_ADDR \
 --env HYPERACTOR_MESH_INDEX=\$HYPERACTOR_MESH_INDEX \
 --ipc=host --network=host --device=/dev/kfd --device=/dev/dri  --cap-add=SYS_PTRACE  --cap-add=CAP_SYS_ADMIN  \
 --security-opt seccomp=unconfined --group-add video --privileged --device=/dev/infiniband \
 -v /mnt/models/mreso/:/mnt/models/mreso/ \
 \${IB_MOUNT_OPTIONS} \
 \$DOCKER_IMAGE /bin/bash"



