import torch
import time
import statistics
import psutil
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

def run_benchmark(model_name, prompt, num_runs=5, max_new_tokens=100, device="cuda"):
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Configure tokenizer if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Move model to device
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded with {total_params/1e6:.2f}M parameters")
    
    # Prepare input
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_token_count = inputs.input_ids.shape[1]
    
    print("Warming up.")
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=max_new_tokens)
    
    # Benchmark runs
    print(f"Running {num_runs} benchmark iterations...")
    generation_times = []
    memory_usages = []
    
    for i in range(num_runs):
        # Clear cache
        if device == "cuda":
            torch.cuda.empty_cache()
        
        # Record starting memory
        if device == "cuda":
            start_mem = torch.cuda.memory_allocated() / 1e9  # Convert to GB
        else:
            start_mem = psutil.Process(os.getpid()).memory_info().rss / 1e9
            
        # Record starting time
        start_time = time.time()
        
        # Generate text
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=max_new_tokens)
        
        # Synchronize if using CUDA
        if device == "cuda":
            torch.cuda.synchronize()
            
        # Record end time
        end_time = time.time()
        
        # Record peak memory
        if device == "cuda":
            end_mem = torch.cuda.memory_allocated() / 1e9
        else:
            end_mem = psutil.Process(os.getpid()).memory_info().rss / 1e9
        
        generation_times.append(end_time - start_time)
        memory_usages.append(max(start_mem, end_mem))
        
        # Decode output
        output_text = tokenizer.decode(output[0], skip_special_tokens=True)
        output_token_count = output.shape[1]
        
        print(f"Run {i+1}: {generation_times[-1]:.2f} seconds, {memory_usages[-1]:.2f} GB peak memory")
    
    avg_time = statistics.mean(generation_times)
    avg_memory = statistics.mean(memory_usages)
    tokens_per_second = (output_token_count - input_token_count) / avg_time
    

    print("\n===== BENCHMARK RESULTS =====")
    print(f"Model: {model_name}")
    print(f"Parameters: {total_params/1e6:.2f}M")
    print(f"Device: {device}")
    print(f"Average generation time: {avg_time:.4f} seconds")
    print(f"Tokens per second: {tokens_per_second:.2f}")
    print(f"Average memory usage: {avg_memory:.2f} GB")
    
    return {
        "model": model_name,
        "params": total_params,
        "avg_time": avg_time,
        "tokens_per_second": tokens_per_second,
        "avg_memory": avg_memory,
        "output_text": output_text
    }

# Run benchmarks
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Test prompt [change as needed]
    prompt = "Explain the concept of Mixture of Experts in neural networks and how it helps with efficiency."
    
    # Run benchmark on Qwen MoE model
    moe_results = run_benchmark(
        model_name="Qwen/Qwen1.5-MoE-A2.7B",
        prompt=prompt,
        device=device
    )
    
    standard_results = run_benchmark(
        model_name="Qwen/Qwen1.5-1.8B",  # A non-MoE Qwen model for comparison
        prompt=prompt,
        device=device
    )
    
    # Compare results
    print("\n===== COMPARISON =====")
    print(f"MoE model ({moe_results['params']/1e6:.2f}M params): {moe_results['tokens_per_second']:.2f} tokens/sec")
    print(f"Standard model ({standard_results['params']/1e6:.2f}M params): {standard_results['tokens_per_second']:.2f} tokens/sec")
    speedup = moe_results['tokens_per_second'] / standard_results['tokens_per_second']
    print(f"Speed ratio: {speedup:.2f}x")