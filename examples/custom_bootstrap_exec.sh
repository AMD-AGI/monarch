#!/bin/bash
# Execute processes inside a Docker container (auto-creates if needed)

# export DOCKER_IMAGE=${DOCKER_IMAGE:-"docker.io/rocm/megatron-lm:v25.5_py310"}
# export DOCKER_IMAGE=${DOCKER_IMAGE:-"docker.io/library/monarch_amd"}

#export DOCKER_IMAGE=${DOCKER_IMAGE:-"docker.io/rocm/megatron-lm:v25.6_py312"}

# IMPORTANT
# This config is important for making sure multi-node test will run, need to configure it carefully depending on the cluster 
export NCCL_IB_HCA=${NCCL_IB_HCA:="bnxt_re0,bnxt_re1,bnxt_re2,bnxt_re3,bnxt_re4,bnxt_re5,bnxt_re7,bnxt_re8"} # modify based on the GPU NiC settings


# Use hostname to identify the container for this node
CONTAINER_NAME="monarch_node_$(hostname)"
echo "[P${HYPERACTOR_MESH_INDEX}] Starting on $(hostname), container: ${CONTAINER_NAME}"

docker ps --format '{{.Names}}'

# Create container if it doesn't exist (with lock to prevent race conditions)
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    (
        flock -x 200
        if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
            echo "[P${HYPERACTOR_MESH_INDEX}] Creating container..."
            /mnt/models/mreso/monarch/examples/init_docker_container.sh || exit 1
        fi
    ) 200>"/tmp/${CONTAINER_NAME}.lock"
fi

echo "[P${HYPERACTOR_MESH_INDEX}] Executing in container"

# Execute inside container
# docker exec \
#  --env HYPERACTOR_MESH_BOOTSTRAP_ADDR=${HYPERACTOR_MESH_BOOTSTRAP_ADDR} \
#  --env HYPERACTOR_MESH_INDEX=${HYPERACTOR_MESH_INDEX} \
#  --env NCCL_DEBUG=INFO \
#  ${CONTAINER_NAME} \
#  /bin/bash -c \
#  "echo \$(date) [Process ${HYPERACTOR_MESH_INDEX}] ; \
#  echo \$(date) [Process ${HYPERACTOR_MESH_INDEX}]: ${HYPERACTOR_MESH_BOOTSTRAP_ADDR} ; \
#   eval \"\$(/mnt/models/mreso/monarch/miniforge3/bin/conda shell.bash hook)\" ; \
#   source /mnt/models/mreso/monarch/miniforge3/etc/profile.d/conda.sh ; \
#   conda activate /mnt/models/mreso/monarch/miniforge3/envs/monarch ; \
#    monarch_bootstrap; \
#   echo \$(date) [P${HYPERACTOR_MESH_INDEX}] Completed"
#    monarch_bootstrap 2>&1 | tee /mnt/models/mreso/monarch/monarch_bootstrap_${HYPERACTOR_MESH_INDEX}.log ; \

# No need to activate external conda enviornment if we use Docker which already has Monarch installed
docker exec \
 --env HYPERACTOR_MESH_BOOTSTRAP_ADDR=${HYPERACTOR_MESH_BOOTSTRAP_ADDR} \
 --env HYPERACTOR_MESH_INDEX=${HYPERACTOR_MESH_INDEX} \
 --env NCCL_DEBUG=INFO \
 --env NCCL_IB_HCA=\$NCCL_IB_HCA \
 ${CONTAINER_NAME} \
 /bin/bash -c \
 "echo \$(date) [Process ${HYPERACTOR_MESH_INDEX}] ; \
 echo \$(date) [Process ${HYPERACTOR_MESH_INDEX}]: ${HYPERACTOR_MESH_BOOTSTRAP_ADDR} ; \
   monarch_bootstrap; \
  echo \$(date) [P${HYPERACTOR_MESH_INDEX}] Completed"




if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    (
        flock -x 200
        if  docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
            echo "[P${HYPERACTOR_MESH_INDEX}] Stopping container..."
            docker stop ${CONTAINER_NAME} || exit 1
        fi
    ) 200>"/tmp/${CONTAINER_NAME}_2.lock"
fi



rm /tmp/${CONTAINER_NAME}.lock
rm /tmp/${CONTAINER_NAME}_2.lock