import fire
import os
from transformers import LlamaForCausalLM, AutoTokenizer, LlamaTokenizer

import torch
import torch.distributed as dist
from torch import nn, Tensor
from torch.distributed._shard.sharded_tensor.api import ShardedTensor
from torch.distributed._tensor import DeviceMesh, DTensor
from torch.distributed.fsdp._fsdp_extensions import (
    _ext_chunk_dtensor,
    _ext_chunk_tensor,
)
import torch.distributed.checkpoint as dist_cp

# command to run
# torchrun --nnodes 1 --nproc_per_node 2 hf_convertor.py --model_name meta-llama/Llama-2-7b-chat-hf --save_checkpoint_dir hf-dtensor-checkpoints


def convert_checkpoint_to_dtensor(state_dict, mesh, save_checkpoint_dir):
    dist_state_dict = {}
    for fqn, tensor in state_dict.items():
        # Hack for buffer
        if "inv_freq" or "cos_cached"  or "sin_cached" in fqn:
            dist_state_dict[fqn] = tensor.clone()
            continue
   
       
        assert mesh is not None
        tensor = _ext_chunk_dtensor(
            tensor=tensor.contiguous(),
            rank=dist.get_rank(),
            device_mesh=mesh,
        )
         
        # try:
        #     if isinstance(tensor, DTensor):
        #         print(f"{fqn} is DTensor")
        # except:
        #     print(f"{fqn} is not DTensor")
        dist_state_dict[fqn] = tensor
    # assert isinstance(tensor, DTensor), f"The tensor at fqn '{fqn}' is not a DTensor."
    dtypes = {v.dtype for v in dist_state_dict.values()}
    print(f"Made dist_state_dict with dtypes {dtypes}")
    dist_cp.save_state_dict(
            state_dict=dist_state_dict,
            storage_writer=dist_cp.FileSystemWriter(save_checkpoint_dir),
        )
    print(f" the DTensor model checkpoints has been saved in{save_checkpoint_dir}")


def main(model_name, save_checkpoint_dir):
    backend = "nccl" 
    dist.init_process_group("nccl")
    _rank = int(os.environ["RANK"])
    _local_rank = int(os.environ["LOCAL_RANK"])

    world_size = int(os.environ["WORLD_SIZE"])  # total number of training processes
    device = f"cuda:{_local_rank}"
    torch.cuda.set_device(device)
    model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    mesh = (
        DeviceMesh(
            device_type="cuda",
            mesh=list(range(dist.get_world_size())),
        ))
    
    convert_checkpoint_to_dtensor(model.state_dict(), mesh, save_checkpoint_dir)

if __name__=="__main__":
    fire.Fire(main)