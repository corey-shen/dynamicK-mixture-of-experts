import json
import csv
import matplotlib.pyplot as plt

def preprocess_results(results):
    results["tokens_per_millisecond"] = results["tokens_per_second"] / 1000
    results["avg_memory"] = results["avg_memory"] * 1024     # GB → MB
    results["avg_time"] = results["avg_time"] * 1000         # sec → ms
    results["efficiency"] = results["tokens_per_millisecond"] / (results["params"] / 1e6)
    return results

def load_benchmark_data(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
        for model in data:
            preprocess_results(model)
        return data

def save_to_csv(results_list, filename, metrics, labels):
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Model", "Parameters (M)"] + labels)
        for res in results_list:
            writer.writerow([
                res["model"],
                res["params"] / 1e6,
                *[res[m] for m in metrics]
            ])

def plot_benchmarks(results, metrics, labels, output_prefix="benchmark_comparison"):
    x = range(len(labels))
    width = 0.35
    plt.figure(figsize=(10, 6))
    for idx, res in enumerate(results):
        offset = (idx - (len(results) - 1)/2) * width
        bars = plt.bar([i + offset for i in x], [res[m] for m in metrics], width=width, label=res["model"])
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f'{yval:.2f}', ha='center', va='bottom', fontsize=8)
    plt.xticks(x, labels, rotation=15)
    plt.ylabel("Metric Value")
    plt.title("Model Benchmark Comparison (MB / ms)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}.png", dpi=300)
    plt.savefig(f"{output_prefix}.pdf", dpi=300)
    plt.show()

# --- Manual MoE vs Standard Results ---
try:
    moe_results = preprocess_results(moe_results)
    standard_results = preprocess_results(standard_results)
    manual_results = [moe_results, standard_results]
except NameError:
    manual_results = []

# Load JSON benchmark results
input_file = "benchmark_results.json"
json_results = load_benchmark_data(input_file)

all_results = manual_results + json_results

# CSV and Plot Configuration
csv_output_file = "benchmark_results_combined.csv"
plot_prefix = "benchmark_comparison_combined"
metrics = ["tokens_per_millisecond", "avg_memory", "avg_time", "latency", "accuracy", "efficiency"]
labels = ["Tokens/ms", "Memory (MB)", "Time (ms)", "Latency (ms)", "Accuracy (%)", "Efficiency"]

# Output Results
save_to_csv(all_results, csv_output_file, metrics, labels)
plot_benchmarks(all_results, metrics, labels, plot_prefix)

print(f"\n Saved combined benchmark results to {csv_output_file} and plotted as '{plot_prefix}.png'")
