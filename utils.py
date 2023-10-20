import torch
from torch.distributed._tensor import DeviceMesh, DTensor
from torch.distributed.tensor.parallel import (
        PairwiseParallel,
        parallelize_module,
        ColwiseParallel,
        RowwiseParallel,
    )
import torch.distributed as dist
import os
from torch._dynamo.utils import CompileProfiler
from hf_convertor import convert_checkpoint_to_dtensor
import torch.distributed.checkpoint as dist_cp

def print_submodules(model):
        for name, module in model.named_modules():
            print(f"Module name: {name}")
            # print(module)
            print()
            
def parallelize_llama_MLP_block(model, module_path, mesh):
    block = model.get_submodule(module_path)
    parallelized_block = parallelize_module(
        module=block,
        device_mesh=mesh,
        parallelize_plan={
            "gate_proj": ColwiseParallel(),
            "up_proj": ColwiseParallel(),
            "down_proj": RowwiseParallel(),
        },
        # tp_mesh_dim=0,
    )
    return parallelized_block

def tp_llama(model, mesh):
    for i in range(model.config.num_hidden_layers):
        block = parallelize_llama_MLP_block(model, f"model.layers.{i}.mlp", mesh)

def _load_tp_checkpoints(tp_model,CHECKPOINT_DIR):
    
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank == -1:
        raise RuntimeError("Expected local_rank to be set, but it is not!")
    tp_state_dict = tp_model.state_dict()
    dist_cp.load_state_dict(
            state_dict=tp_state_dict,
            storage_reader=dist_cp.FileSystemReader(CHECKPOINT_DIR),
        )
    tp_model.load_state_dict(tp_state_dict)
    