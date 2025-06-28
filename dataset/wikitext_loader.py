from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import torch

# keep in mind that split can be “train” or “validation” or “test”, and the seq_len is the model context length
def get_wikitext103(split="train", seq_len=1024, batch_size=4): # optimize/load balancing seq_len via number of heads running
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_set = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)

    # tokenizes the sample, turning raw strings into input token IDs.
    tokenizer=AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    def tokenize(sample):
        print(tokenizer)
        # print(tokenizer(sample["text"]))
        ids = tokenizer(sample["text"]).input_ids
        # you can drop empty lines
        return {"ids": [i for i in ids if i]}

    tokenized = data_set.map(tokenize, batched=False, remove_columns=["text"])

    # this is where you chunk everything into fixed windows
    def chunk(example):
        ids = example["ids"]
        out = [ids[i : i + seq_len]       # sliding windows
               for i in range(0, len(ids) - seq_len, seq_len)]
        print(f"IDS: {ids}")
        print(f"Out: {out}")
        return {"input_ids": out}

    windows = tokenized.map(chunk, batched=False, remove_columns=["ids"])
    windows.set_format(type="torch")

    # DataLoader that yields dict(input_ids, attention_mask) batches.
    return DataLoader(windows, batch_size=batch_size, shuffle=(split == "train"))
get_wikitext103("train", 1024, 4)