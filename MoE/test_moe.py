import torch
from MoE import DynamicMoE, MoE

def test_dynamic_moe():
    # Same code as mixture_of_experts.py
    return 0

def benchmark_performance():
    import time
    configs = [
        {"tau": 0.5, "max_k":2},
        {"tau": 0.8, "max_k":3},
        {"tau": 0.95, "max_k":4}
    ]

    for config in configs:
        model = DynamicMoE(dim=512, num_experts = 16, **config)
        inputs = torch.randn(4, 64, 512)

        start_time = time.time()
        with torch.no_grad():
            output, loss = model(inputs)
        end_time = time.time()

        print(f"Config {config}: {end_time - start_time:.4f}s, Loss: {loss.item():.6f}")

if __name__ == "__main__":
    test_dynamic_moe()
    benchmark_performance()
