import torch, math
from mixture_of_experts.mixture_of_experts import test_tau_sensitivity
from wikitext_loader import get_wikitext103
from transformers import AutoTokenizer

model_id  = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = DynamicMoE.load_pretrained(model_id).cuda().eval()

dl = get_wikitext103("test", seq_len=1024, batch_size=4, tokenizer=tokenizer)

total_loss, n_tokens = 0.0, 0
ce = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")

with torch.no_grad():
    for batch in dl:
        ids = batch["input_ids"].cuda()
        labels = ids.clone()
        logits = model(ids).logits          # (b, t, vocab)
        loss   = ce(logits[:, :-1].reshape(-1, logits.size(-1)),
                    labels[:, 1:].reshape(-1))
        total_loss += loss.item()
        n_tokens   += ids[:, 1:].numel()

perplexity = math.exp(total_loss / n_tokens)
print(f"Perplexity Score on WikiText-103: {perplexity:8.2f}")
