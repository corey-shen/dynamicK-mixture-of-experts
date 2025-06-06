import time
import torch
import psutil
import json
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM


def get_device():
    if torch.cuda.is_available():
        print("Using CUDA")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("Using MPS")
        return torch.device("mps")
    else:
        print("Using CPU")
        return torch.device("cpu")


class ModelBenchmark:
    def __init__(self, model_name):
        self.model_name = model_name
        self.device = get_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # Fix attention mask warning by setting pad_token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token

        self.metrics = {}

    def warmup(self):
        print(f"Warming up {self.model_name} ...")
        inputs = self.tokenizer("Hello world", return_tensors="pt").to(self.device)
        with torch.no_grad():
            self.model.generate(**inputs, max_new_tokens=5)

    def run_benchmark(self, prompt="What is machine learning?", max_new_tokens=50):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        input_len = input_ids.shape[1]

        torch.cuda.empty_cache() if self.device.type == "cuda" else None

        start_time = time.perf_counter()
        with torch.no_grad():
            output = self.model.generate(input_ids, max_new_tokens=max_new_tokens)
        end_time = time.perf_counter()

        elapsed = end_time - start_time
        output_len = output.shape[1]
        new_tokens = output_len - input_len
        tokens_per_sec = new_tokens / elapsed if elapsed > 0 else 0

        # Memory usage (approximate) in MB
        process = psutil.Process()
        mem = process.memory_info().rss / (1024 * 1024)

        self.metrics = {
            "model": self.model_name,
            "compute_time": elapsed,
            "tokens_per_second": tokens_per_sec,
            "memory_mb": mem,
            "params_million": sum(p.numel() for p in self.model.parameters()) / 1e6,
        }
        print(f"Done {self.model_name}: {self.metrics}")
        return self.metrics


def plot_results(results):
    import numpy as np

    labels = ["Tokens/sec", "Memory (MB)", "Time (s)"]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, res in enumerate(results):
        values = [res["tokens_per_second"], res["memory_mb"], res["compute_time"]]
        offset = (i - len(results)/2) * width + width/2
        bars = ax.bar(x + offset, values, width, label=res["model"])
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + 0.05, f"{yval:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Model Benchmark Comparison")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def main():
    # Replace these with your actual model repo IDs or paths
    dense_model_name = "meta-llama/Llama-3.2-1B"  # dense model (gated repo, requires token)
    benchmark_model_name = "gpt2"  # example small open model for testing

    # Run benchmarks
    dense_benchmark = ModelBenchmark(dense_model_name)
    dense_benchmark.warmup()
    dense_results = dense_benchmark.run_benchmark()

    benchmark_benchmark = ModelBenchmark(benchmark_model_name)
    benchmark_benchmark.warmup()
    benchmark_results = benchmark_benchmark.run_benchmark()

    # Save results
    all_results = [dense_results, benchmark_results]
    with open("benchmark_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Plot results
    plot_results(all_results)


if __name__ == "__main__":
    main()
