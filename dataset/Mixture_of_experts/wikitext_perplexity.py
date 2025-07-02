import torch, math
print("ABHAHAFHFHFHA")
from mixture_of_experts import DynamicMoE
print("XXXXXX")
# from wikitext_loader import get_wikitext103
from transformers import AutoTokenizer
print("YYYYYYYY")
from datasets import load_from_disk
print("ZZZZZZZZZ")

model_id  = "Qwen/Qwen3-4B"

tokenizer = load_from_disk("tokenized_wikitext103")
model = DynamicMoE.load_pretrained(model_id).cuda().eval()

# dl = get_wikitext103("train", seq_len=1024, batch_size=4, tokenizer=tokenizer)

total_loss, n_tokens = 0.0, 0
ce = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")

with torch.no_grad():
    for batch in tokenizer:
        ids = batch["input_ids"].cuda()
        labels = ids.clone()
        logits = model(ids).logits          # (b, t, vocab)
        loss   = ce(logits[:, :-1].reshape(-1, logits.size(-1)),
                    labels[:, 1:].reshape(-1))
        total_loss += loss.item()
        n_tokens   += ids[:, 1:].numel()

perplexity = math.exp(total_loss / n_tokens)
print(f"Perplexity Score on WikiText-103: {perplexity:8.2f}")
