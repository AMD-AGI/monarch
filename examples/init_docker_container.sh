#!/bin/bash
# This script initializes one Docker container per node
# It should be called ONCE per node before spawning processes

CONTAINER_MOUNT=${MONARCH_EXAMPLE_FOLDER}/../../

export NCCL_IB_DISABLE=1

export NCCL_TIMEOUT=600  

export NCCL_P2P_LEVEL=SYS  

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

NCCL_DEBUG=INFO 
export DOCKER_IMAGE=${DOCKER_IMAGE:-"docker.io/library/monarch_amd"}

# Use hostname to create unique container name per node
CONTAINER_NAME="monarch_node_$(hostname)_${USER}"

echo "[INIT] Starting Docker container initialization on node $(hostname)"
echo "[INIT] Container name: ${CONTAINER_NAME}"

# Force cleanup: remove ALL existing containers on this node
echo "[INIT] Force cleaning up ALL existing containers on $(hostname)"
docker ps -aq | xargs -r docker rm -f 2>/dev/null || true
echo "[INIT] All old containers removed"

echo "[INIT] Creating new container ${CONTAINER_NAME}"

echo "MONARCH_EXAMPLE_FOLDER=${MONARCH_EXAMPLE_FOLDER}"

# Create new container in detached mode to keep it running
if docker run --rm -d --name ${CONTAINER_NAME} \
 --env NCCL_IB_HCA=${NCCL_IB_HCA} \
 --env NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME} \
 --env CONTAINER_MOUNT=${CONTAINER_MOUNT} \
 --env NCCL_DEBUG=${NCCL_DEBUG} \
 --ipc=host --network=host --device=/dev/kfd --device=/dev/dri \
 --cap-add=SYS_PTRACE --cap-add=CAP_SYS_ADMIN \
 --security-opt seccomp=unconfined --group-add video --privileged \
 --device=/dev/infiniband \
 -v ${CONTAINER_MOUNT}:${CONTAINER_MOUNT} \
 -v /etc/hosts:/etc/hosts \
 ${DOCKER_IMAGE} \
 /bin/bash -c "tail -f /dev/null"; then
    
    echo "[INIT] Container ${CONTAINER_NAME} created successfully"
    echo "[INIT] Waiting for container to be fully ready..."
    sleep 3
    
    echo "[INIT] Container ${CONTAINER_NAME} is ready for processes"
    exit 0
else
    echo "[INIT] ERROR: Failed to create container ${CONTAINER_NAME}"
    exit 1
fi

#  ${IB_MOUNT_OPTIONS} \