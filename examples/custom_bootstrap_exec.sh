#!/bin/bash
# Execute processes inside a Docker container (auto-creates if needed)

# IMPORTANT
# This config is important for making sure multi-node test will run, need to configure it carefully depending on the cluster 
export NCCL_IB_HCA=${NCCL_IB_HCA:="bnxt_re0,bnxt_re1,bnxt_re2,bnxt_re3,bnxt_re4,bnxt_re5,bnxt_re7,bnxt_re8"} # modify based on the GPU NiC settings

# Use hostname to identify the container for this node
CONTAINER_NAME="monarch_node_$(hostname)_${USER}"

echo "[P${HYPERACTOR_MESH_INDEX}] Starting on $(hostname), container: ${CONTAINER_NAME}"


docker ps --format '{{.Names}}'

# Create container if it doesn't exist (with lock to prevent race conditions)
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    (
        flock -x 200
        if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
            echo "[P${HYPERACTOR_MESH_INDEX}] Creating container..."
            ${MONARCH_EXAMPLE_FOLDER}/init_docker_container.sh || exit 1
        fi
    ) 200>"/tmp/${CONTAINER_NAME}_${USER}.lock"
fi

echo "[P${HYPERACTOR_MESH_INDEX}] Executing in container"

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
 echo '$2' > /tmp/custom_bootstrap_exec.py ;
 LD_LIBRARY_PATH=/opt/ompi/lib:/opt/rocm/lib:/usr/local/lib::/opt/rocm/lib/:/usr/lib/x86_64-linux-gnu/ python /tmp/custom_bootstrap_exec.py
 echo \$(date) [P${HYPERACTOR_MESH_INDEX}] Completed"


if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    (
        flock -x 200
        if  docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
            echo "[P${HYPERACTOR_MESH_INDEX}] Stopping container..."
            docker stop ${CONTAINER_NAME} || exit 1
        fi
    ) 200>"/tmp/${CONTAINER_NAME}_${USER}_2.lock"
fi


rm /tmp/${CONTAINER_NAME}_${USER}.lock
rm /tmp/${CONTAINER_NAME}_${USER}_2.lock