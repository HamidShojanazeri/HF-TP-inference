# HF-TP-inference
PyTorch Native Tensor Parallel for HuggingFace models inference


### Run the inference TP(lized) model +Compile

Example of HF llama 7B

```bash
torchrun --nnodes 1 --nproc_per_node 2 llama-simple.py --model_name meta-llama/Llama-2-7b-chat-hf --compile

```
### Convert HF checkpoints to DTensor Checkpoints


```bash
torchrun --nnodes 1 --nproc_per_node 2 hf_convertor.py --model_name meta-llama/Llama-2-7b-chat-hf --save_checkpoint_dir hf-dtensor-checkpoints
```

### Run the inference with deferred init TP(lized) model +compile


```bash
torchrun --nnodes 1 --nproc_per_node 2 llama-simple.py --model_name meta-llama/Llama-2-7b-chat-hf --checkpoint_dir hf-dtensor-checkpoints --compile --meta_device

```