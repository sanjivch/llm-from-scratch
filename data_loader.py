import torch
from torch.utils.data import Dataset, DataLoader

class LLMDataset(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.in_token_ids = []
        self.out_token_ids = []

        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids)-max_length, stride):
            in_chunk = token_ids[i:i+max_length]
            out_chunk = token_ids[i+1: i+max_length+1]
            self.in_token_ids.append(torch.tensor(in_chunk))
            self.out_token_ids.append(torch.tensor(out_chunk))

    def __len__(self) -> int:
        return len(self.in_token_ids)

    def __getitem__(self, idx: int):
        return self.in_token_ids[idx],self.out_token_ids[idx]

