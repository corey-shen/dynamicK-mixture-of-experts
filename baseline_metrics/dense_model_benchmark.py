from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, time, json, os, psutil, math
from datasets import load_dataset
from torchsummary import summary
import matplotlib.pyplot as plt

def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using CUDA acceleration")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS acceleration")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device

try:
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    exit()

device = get_device()
model = model.to(device)
model_name = model.config.name_or_path

class ModelBenchmark:
    def __init__(self, model, tokenizer, device, model_name):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model_name = model_name
        self.compute_time = 0.0
        self.tokens_per_second_val = 0.0
        self.throughput_val = 0.0
        self.utilization_val = 0.0
        self.perplexity_score = None

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token

    def warmup(self, prompt="Hello", max_new_tokens=5):
        print("Running warmup to reduce overhead...")
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            self.model.generate(**inputs, max_new_tokens=max_new_tokens)

    def run_inference(self, prompt, max_new_tokens=500):
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = inputs.to(self.device)
        input_token_count = inputs['input_ids'].shape[1]

        if self.device.type == 'cuda':
            self.compute_time = self.time_cuda_inference(inputs['input_ids'], max_new_tokens)
        else:
            self.compute_time = self.time_cpu_inference(inputs['input_ids'], max_new_tokens)

        output = self.model.generate(
            inputs['input_ids'],
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=self.tokenizer.pad_token_id
        )
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)

        output_token_count = output.shape[1]
        new_tokens = output_token_count - input_token_count
        self.tokens_per_second_val = new_tokens / self.compute_time if self.compute_time > 0 else 0
        self.throughput_val = output_token_count / self.compute_time if self.compute_time > 0 else 0
        self.utilization_val = self.compute_utilization()

        return response

    def time_cuda_inference(self, input_ids, max_new_tokens):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.no_grad():
            start.record()
            self.model.generate(input_ids, max_new_tokens=max_new_tokens)
            end.record()
            torch.cuda.synchronize()
        elapsed_time_ms = start.elapsed_time(end)
        return elapsed_time_ms / 1000

    def time_cpu_inference(self, input_ids, max_new_tokens):
        start_time = time.perf_counter()
        with torch.no_grad():
            self.model.generate(input_ids, max_new_tokens=max_new_tokens)
        end_time = time.perf_counter()
        return end_time - start_time

    def flops_per_second(self):
        params = sum(p.numel() for p in self.model.parameters())
        return (params * 2) / self.compute_time if self.compute_time > 0 else 0

    def compute_utilization(self):
        total_mem = psutil.virtual_memory().total
        used_mem = psutil.Process().memory_info().rss
        return used_mem / total_mem

    def compute_perplexity_on_wikitext(self, num_samples=1000, max_length=512):
        print("Loading WikiText-103...")
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
        texts = [t for t in dataset["test"]["text"] if len(t.strip()) > 0][:]

        total_loss = 0.0
        total_tokens = 0

        print("Calculating perplexity...")
        for text in texts:
            enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            input_ids = enc.input_ids.to(self.device)
            labels = input_ids.clone()

            with torch.no_grad():
                outputs = self.model(input_ids, labels=labels)
                loss = outputs.loss
                total_loss += loss.item() * input_ids.numel()
                total_tokens += input_ids.numel()

        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        self.perplexity_score = perplexity
        print(f"Perplexity on WikiText-103 (subset of {num_samples} samples): {perplexity:.2f}")
        return perplexity

    def save_metrics_to_json(self):
        metrics = {
            "Model Name": self.model_name,
            "Device": str(self.device),
            "Compute Time (s)": self.compute_time,
            "Tokens/sec": self.tokens_per_second_val,
            "FLOPs/sec": self.flops_per_second(),
            "Throughput (tokens/s)": self.throughput_val,
            "Compute Utilization": self.utilization_val,
            "Memory (MB)": psutil.Process().memory_info().rss / (1024 * 1024),
            "Perplexity (WikiText-103)": self.perplexity_score if self.perplexity_score is not None else "N/A"
        }
        project_root = os.getcwd()
        storage_dir = os.path.join(project_root, "baseline_benchmark_results")
        os.makedirs(storage_dir, exist_ok=True)
        file_path = os.path.join(storage_dir, "benchmark_results.json")

        with open(file_path, "w") as f:
            json.dump(metrics, f, indent=4)

        print(f"Saved benchmark results to: {file_path}")

# -------- RUNNING BENCHMARK --------

benchmark = ModelBenchmark(model, tokenizer, device, model_name)
benchmark.warmup()

print("\n=== Generated Text ===")
prompt = "The history of artificial intelligence dates back to"
print(benchmark.run_inference(prompt))

# Evaluate perplexity on WikiText-103 (subset)
benchmark.compute_perplexity_on_wikitext(num_samples=500)

# Save full results
benchmark.save_metrics_to_json()