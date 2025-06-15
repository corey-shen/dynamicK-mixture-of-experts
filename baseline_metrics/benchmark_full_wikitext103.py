import os
import time
import math
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from fvcore.nn import FlopCountAnalysis
from datasets import load_dataset

# Optional GPU utilization via pynvml
try:
    from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates
    NVML_AVAILABLE = True
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
except Exception:
    NVML_AVAILABLE = False


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def measure_flops_per_token(model, seq_len, batch_size, device):
    model_cpu = model.to('cpu').eval()
    # dummy input
    dummy_ids = torch.randint(0, 1000, (batch_size, seq_len), dtype=torch.long)
    macs = FlopCountAnalysis(model_cpu, (dummy_ids,))
    total_mac = macs.total()
    total_flops = total_mac * 2
    flops_per_token = total_flops / seq_len
    return flops_per_token


def measure_performance(model, tokenizer, prompt, seq_len, max_new_tokens, batch_size, num_runs, device):
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True,
                       padding='max_length', max_length=seq_len).to(device)
    if batch_size > 1:
        inputs = {k: v.repeat(batch_size, 1) for k, v in inputs.items()}
    model.eval().to(device)

    # Warm-up
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=max_new_tokens)
    torch.cuda.synchronize()

    times, mems, utils = [], [], []
    output_text = None
    for i in range(num_runs):
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens)
        torch.cuda.synchronize()
        t1 = time.time()
        times.append(t1 - t0)
        mems.append(torch.cuda.max_memory_allocated() / 1e9)
        if NVML_AVAILABLE:
            util = nvmlDeviceGetUtilizationRates(handle).gpu
            utils.append(util)
        if output_text is None:
            output_text = tokenizer.decode(out[0], skip_special_tokens=True)
    avg_time = sum(times) / len(times)
    tokens_per_sec = ((inputs['input_ids'].shape[1] + max_new_tokens) * batch_size) / avg_time
    avg_mem = sum(mems) / len(mems)
    avg_util = sum(utils) / len(utils) if utils else None
    return avg_time, tokens_per_sec, avg_mem, avg_util, output_text


def measure_expert_load(model, tokenizer, prompt, seq_len, max_new_tokens, batch_size, device, k=2):
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True,
                       padding='max_length', max_length=seq_len).to(device)
    if batch_size > 1:
        inputs = {k: v.repeat(batch_size, 1) for k, v in inputs.items()}
    model.eval().to(device)
    # find router
    routers = [m for n, m in model.named_modules() if n.endswith('.mlp.gate')]
    assert routers, "No router found"
    router = routers[0]
    # count experts
    try:
        E = router.out_features
    except:
        E = next(p.shape[0] for p in router.parameters() if p.ndim == 2)
    counts = np.zeros(E, dtype=int)
    k_eff = min(k, E)

    def hook(module, inp, out):
        logits = out if isinstance(out, torch.Tensor) else out[0]
        probs = F.softmax(logits, dim=-1)
        _, idx = torch.topk(probs, k_eff, dim=-1)
        flat = idx.detach().cpu().numpy().ravel()
        for ex in flat:
            counts[ex] += 1

    handle_hook = router.register_forward_hook(hook)
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=max_new_tokens)
    torch.cuda.synchronize()
    handle_hook.remove()

    mean = counts.mean()
    std = counts.std()
    mx = counts.max()
    total_tokens = inputs['input_ids'].shape[0] * inputs['input_ids'].shape[1]
    capacity = math.ceil(total_tokens / E)
    overflow = int(np.sum(np.maximum(counts - capacity, 0)))
    return mean, std, mx, capacity, overflow


def compute_next_token_accuracy(model, tokenizer, val_file, seq_len, device, batch_size=8):
    total_correct = 0
    total = 0
    model.eval().to(device)
    # read lines
    with open(val_file, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]
    for i in range(0, len(lines), batch_size):
        batch = lines[i:i+batch_size]
        enc = tokenizer(batch, return_tensors='pt', truncation=True,
                        padding='max_length', max_length=seq_len)
        ids = enc['input_ids'].to(device)
        att = enc['attention_mask'].to(device)
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=att, labels=ids)
            logits = out.logits
        preds = logits.argmax(dim=-1)
        for b in range(preds.size(0)):
            real = att[b].sum().item()
            for j in range(real-1):
                total += 1
                if preds[b,j] == ids[b,j+1]:
                    total_correct += 1
    return total_correct / total if total>0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--seq_len', type=int, default=32)
    parser.add_argument('--max_new_tokens', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--runs', type=int, default=50)
    parser.add_argument('--val_file', required=True)
    parser.add_argument('--prompt', type=str, default="The quick brown fox jumps over the lazy dog.")
    args = parser.parse_args()

    device = get_device()
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    mdl = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16 if device.type=='cuda' else torch.float32,
        trust_remote_code=True
    ).to(device)

    print("Model:", args.model_path)
    print(f"Parameters: {count_params(mdl)/1e9:.3f} B")

    flop = measure_flops_per_token(mdl, args.seq_len, args.batch_size, device)
    print(f"FLOPs/token: {flop/1e9:.3f} G")

    atime, tps, amem, autil, outtxt = measure_performance(
        mdl, tok, args.prompt, args.seq_len, args.max_new_tokens,
        args.batch_size, args.runs, device
    )
    print(f"Avg gen time: {atime:.4f} s")
    print(f"Tokens/sec: {tps:.2f}")
    print(f"Avg peak memory: {amem:.2f} GB")
    if autil is not None:
        print(f"Avg GPU util: {autil:.1f}%")
    print("Sample output text:", outtxt[:200].replace("\n"," "))

    mean, std, mx, cap, ov = measure_expert_load(
        mdl, tok, args.prompt, args.seq_len, args.max_new_tokens,
        args.batch_size, device, k=2
    )
    print(f"Expert load (mean±std): {mean:.2f} ± {std:.2f}")
    print(f"Max tokens/expert: {mx}")
    print(f"Capacity/head: {cap}")
    print(f"Overflow count: {ov}")

    acc = compute_next_token_accuracy(
        mdl, tok, args.val_file, args.seq_len, device
    )
    print(f"Next-token accuracy: {acc*100:.2f}%")

if __name__ == '__main__':
    main()
