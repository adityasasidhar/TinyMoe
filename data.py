import json

import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class CosmopediaDataset(Dataset):
    def __init__(self, path: str, tokenizer, seq_len: int):
        self.samples = []
        pad_token = tokenizer.pad_token_id

        print("Tokenizing dataset into memory...")
        with open(path, "r") as f:
            for line in tqdm(f):
                data = json.loads(line)
                text = data["text"].strip()
                if len(text) < 20:
                    continue
                ids = tokenizer.encode(text, add_special_tokens=True)
                ids = ids[:seq_len + 1]
                if len(ids) < seq_len + 1:
                    ids += [pad_token] * (seq_len + 1 - len(ids))
                self.samples.append(ids)
        print(f"Loaded {len(self.samples)} padded samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids = self.samples[idx]
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        return x, y
