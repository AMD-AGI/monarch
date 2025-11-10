import torch
from monarch.actor import Actor, endpoint, this_host
from monarch.rdma import RDMABuffer

from monarch.rdma import is_rdma_available
print("RDMA available?", is_rdma_available())

class ParameterServer(Actor):
    def __init__(self):
        self.weights = torch.rand(1000, 1000)
        self.weight_buffer = RDMABuffer(self.weights.view(torch.uint8).flatten())

    @endpoint
    def get_weights(self) -> RDMABuffer:
        print("[server] returning weights")
        return self.weight_buffer

class Worker(Actor):
    def __init__(self):
        self.local_weights = torch.zeros(1000, 1000)

    @endpoint
    def sync_weights(self, server: ParameterServer):
        print("[worker] requesting weights from server")
        weight_ref = server.get_weights.call_one().get()
        print("[worker] copying remote weights into local buffer")
        weight_ref.read_into(self.local_weights.view(torch.uint8).flatten()).get()
        print("[worker] done copying")

server_proc = this_host().spawn_procs(per_host={"gpus": 1})
worker_proc = this_host().spawn_procs(per_host={"gpus": 1})

server = server_proc.spawn("server", ParameterServer)
worker = worker_proc.spawn("worker", Worker)

worker.sync_weights.call_one(server).get()