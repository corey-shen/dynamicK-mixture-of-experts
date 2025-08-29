# Dynamic-k Expert Selection for Mixture of Experts

This repository contains the implementation of **Dynamic-k Expert Selection**, a confidence-aware routing mechanism for Mixture of Experts (MoE) architectures. Unlike static top-k routing, our method dynamically adjusts the number of experts assigned to each token based on router confidence, improving efficiency, reducing wasted computation, and mitigating load imbalance.

This codebase was developed as part of our NeurIPS 2025 workshop submission: *Dynamic k-Expert Selection for Mixture of Expert Architectures*.

## Key Idea
- **Traditional MoEs**: Use a fixed top-k routing strategy (e.g., top-2 experts per token).  
- **Problem**:  
  - Confident tokens are routed to unnecessary extra experts (**wasted compute**).  
  - Ambiguous tokens are limited by fixed-k (**underutilization**).  
- **Our Approach**: Introduce a **confidence threshold `τ`** over softmax probabilities. For each token:
  - Route to the minimal number of experts `k*` such that cumulative probability ≥ `τ`.  
  - Confident tokens → fewer experts.  
  - Ambiguous tokens → more experts.  

This guarantees a **unique minimal k** and balances efficiency with accuracy.

## Features
- **Dynamic-k Gating**: Adaptive expert selection via cumulative probability thresholds.  
- **Custom MoE Layer**: Implements dispatch/combine tensors with auxiliary load-balancing loss.  
- **Language Model Integration**: Lightweight GPT-style transformer backbone with MoE layers.  
- **Dataset Loader**: Supports the [One Billion Word Benchmark](https://arxiv.org/abs/1312.3005), with fallback to dummy data.  
- **Evaluation Metrics**: Perplexity and auxiliary load-balancing loss.  
- **Configurable Thresholds**: We use `τ ∈ {0.5, 0.65, 0.8, 0.95}` to analyze trade-offs.  


## 📊 Results (Perplexity Overview)

| Threshold (τ) | Avg. Perplexity (last 50 epochs) | 95% CI           |
|---------------|----------------------------------|------------------|
| 0.50          | 241.91                           | [239.99, 240.33] |
| 0.65          | 239.95                           | [238.76, 239.11] |
| 0.80          | 239.76                           | [238.26, 238.77] |
| 0.95          | **235.94**                       | [233.39, 234.52] |

- Higher thresholds (e.g., `τ = 0.95`) yield the best perplexity.  
- Extremely low or intermediate thresholds underperform.  
- Dynamic routing reduces wasted allocation while preserving accuracy.  

## Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0
- HuggingFace Transformers
- NumPy, tqdm

## Dataset

This project uses the **One Billion Word Benchmark**.

**Download and extract:**

~~~bash
wget http://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz
tar -xvzf 1-billion-word-language-modeling-benchmark-r13output.tar.gz
~~~

**Place it at:**

~~~text
dynamicK-mixture-of-experts/1-billion-word-language-modeling-benchmark/
~~~

If `data_path` is missing, the loader automatically generates dummy data for a quick demo (smaller scale, not for final results).

## Usage

Run training and evaluation:

~~~bash
python main.py
~~~

**Configuration (inside `main()`):**

~~~python
config = {
    'data_path': '1-billion-word-language-modeling-benchmark',
    'batch_size': 8,
    'learning_rate': 1e-3,
    'num_epochs': 10,
    'max_seq_len': 64,
    'vocab_size': 50257,
    'dim': 256,
    'num_layers': 2,
    'num_experts': 4,
    'num_heads': 4,
    'threshold': 0.8,
    'max_train_samples': 100000,
    'max_test_samples': 100000,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}
~~~

- Tune `threshold` to control sparsity/expert utilization.
- `tokenizer.pad_token` is set to the GPT-2 `eos_token` for padding.
- For quick tests, reduce `max_train_samples` / `max_test_samples`.

## What the Script Prints

- **Per-epoch:** train loss, train perplexity, auxiliary loss
- **Per-epoch eval:** test loss and test perplexity
- **Final:** overall test loss and perplexity

**Example:**

~~~text
Epoch 1 Summary:
  Train Loss: 5.8921
  Train Perplexity: 362.14
  Aux Loss: 0.000012

Evaluating epoch 1...
  Testing Loss: 5.7110
  Testing Perplexity: 302.15
~~~

##  Components Overview

- **Experts:** batched feed-forward experts with GELU activation
- **DynamicKGating:** computes softmax over experts, sorts, cumulative sum vs. τ, builds dispatch and combine tensors with capacity control and aux loss
- **MoE:** wraps gating + experts; returns `(output, aux_loss)`
- **MoELanguageModel:** GPT-style transformer with attention → MoE → residual/LayerNorm
- **OneBillionWordDataset:** loads OBW files or generates dummy samples; produces `(input_ids, targets)` pairs
- **Training loop:** `train_epoch`, `calculate_perplexity`, gradient clipping, AdamW, early batch limits for fast iteration



## Tips & Troubleshooting

- **“Data path does not exist”:** Ensure the dataset folder is exactly `1-billion-word-language-modeling-benchmark/` at the repo root, or change `config['data_path']`.
- **CPU is slow:** Use a CUDA-enabled GPU if available (`'cuda' if torch.cuda.is_available()`).
- **OOM:** Reduce `batch_size`, `max_seq_len`, or `dim`. You can also lower `num_layers` / `num_experts`.
- **Imbalanced experts:** Increase auxiliary loss weight (`loss_coef` in MoE) or tune capacity factors.

## Limitations

- Routing computes scores for all experts, which can add latency in very large MoEs.
- Threshold τ still needs tuning (though results are generally robust).
- Dynamic routing can cause temporary compute/memory spikes.
- **Future work:** lighter routing, adaptive thresholds, and multimodal extensions.


