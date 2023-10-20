# HF-TP-inference
PyTorch Native Tensor Parallel for HuggingFace models inference

### Convert HF checkpoints to DTensor Checkpoints

Example of llama 7B

```bash
torchrun --nnodes 1 --nproc_per_node 2 hf_convertor.py --model_name meta-llama/Llama-2-7b-chat-hf --save_checkpoint_dir hf-dtensor-checkpoints
```

### Run the inference with TP+compile


```bash
torchrun --nnodes 1 --nproc_per_node 2 llama-simple.py --model_name meta-llama/Llama-2-7b-chat-hf --checkpoint_dir hf-dtensor-checkpoints

```