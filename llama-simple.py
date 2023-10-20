import fire

from transformers import LlamaForCausalLM, AutoTokenizer, LlamaTokenizer
# from optimum.bettertransformer import BetterTransformer
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
from utils import print_submodules, parallelize_llama_MLP_block, tp_llama, _load_tp_checkpoints
#command to run 

# torchrun --nnodes 1 --nproc_per_node 2 llama-simple.py 

# def print_submodules(model):
#         for name, module in model.named_modules():
#             print(f"Module name: {name}")
#             # print(module)
#             print()
            
# def parallelize_llama_MLP_block(model, module_path, mesh):
#     block = model.get_submodule(module_path)
#     parallelized_block = parallelize_module(
#         module=block,
#         device_mesh=mesh,
#         parallelize_plan={
#             "gate_proj": ColwiseParallel(),
#             "up_proj": ColwiseParallel(),
#             "down_proj": RowwiseParallel(),
#         },
#         # tp_mesh_dim=0,
#     )
#     return parallelized_block

# def tp_llama(model, mesh):
#     for i in range(model.config.num_hidden_layers):
#         block = parallelize_llama_MLP_block(model, f"model.layers.{i}.mlp", mesh)

# def _load_tp_checkpoints(tp_model,CHECKPOINT_DIR):
    
#     local_rank = int(os.environ.get("LOCAL_RANK", -1))
#     if local_rank == -1:
#         raise RuntimeError("Expected local_rank to be set, but it is not!")
#     tp_state_dict = tp_model.state_dict()
#     dist_cp.load_state_dict(
#             state_dict=tp_state_dict,
#             storage_reader=dist_cp.FileSystemReader(CHECKPOINT_DIR),
#         )
#     tp_model.load_state_dict(tp_state_dict)
    

    

# print_submodules(model)

def main (model_name: str= "meta-llama/Llama-2-7b-chat-hf", checkpoint_dir: str ="hf-dtensor-checkpoints"):
    backend = "nccl" 
    dist.init_process_group("nccl")
    _rank = int(os.environ["RANK"])
    _local_rank = int(os.environ["LOCAL_RANK"])

    world_size = int(os.environ["WORLD_SIZE"])  # total number of training processes
    device = f"cuda:{_local_rank}"

    torch.cuda.set_device(device)
            
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    with torch.device("meta"):
        model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    # model.to(device)
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    # model = BetterTransformer.transform(model)
    mesh = (
            DeviceMesh(
                device_type="cuda",
                mesh=list(range(dist.get_world_size())),
            ))
            
    tp_llama(model, mesh)
    model.to_empty(device='cuda')
    _load_tp_checkpoints(model,"./hf-dtensor-checkpoints")   
    dummy_input = "what is the recipe of ketchup?"
    dummy_input = tokenizer(dummy_input, return_tensors="pt").to(device)

    with CompileProfiler() as prof:
        compiled_model = torch.compile(model, backend="inductor")
        output = compiled_model.generate(
                    **dummy_input,
                    max_new_tokens=20,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.6)
        print(tokenizer.decode(output[0], skip_special_tokens=True))
        print("====================================================")
        print(prof.report())
        
        
if __name__=="__main__":
    fire.Fire(main)

   
        
