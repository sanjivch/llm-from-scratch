import urllib.request
import json
import re
from tokenizer import NaiveTokenizer
import tiktoken
from torch.utils.data import Dataset, DataLoader
from data_loader import LLMDataset

def download_data(url: str, file_path: str):
    urllib.request.urlretrieve(url, file_path)
    return


def create_vocab(file_name: str) -> dict:

    # Read text from the file
    with open("the-verdict.txt", "r") as f:
        raw_text = f.read()

    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    preprocessed = [i.strip() for i in preprocessed if i.strip()]

    tokens = sorted(list(set(preprocessed)))

    # Add special tokens
    special_tokens = ["<|endoftext|>", "<|unk|>"]
    tokens.extend(special_tokens)
    # print(len(tokens), tokens[-1])
    return {token:token_id for token_id, token in enumerate(tokens)}

    
def prepare_data(token_ids: list, context_length: int = 8):
    x = token_ids[:context_length]
    y = token_ids[1:context_length+1]


def create_dataloader(txt:str, batch_size:int = 4, max_length:int = 256, stride:int =128, shuffle:bool = True, drop_last:bool =True) -> DataLoader:
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = LLMDataset(txt, tokenizer=tokenizer, max_length=max_length, stride=stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=0)

    return dataloader



url = ("https://raw.githubusercontent.com/rasbt/"
    "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
    "the-verdict.txt")
file_path = "the-verdict.txt"

# Download file
# download_data(url, file_path)

# Create a vocab and convert to json
vocab = create_vocab(file_name=file_path)
# with open("vocab.json", "w") as f: 
#     json.dump(vocab, f)

# tokenizer = NaiveTokenizer(vocab)
tokenizer = tiktoken.get_encoding("gpt2")
prompt =  """"Hello, do you like tea? <|endoftext|> In the sunlit terraces"
     "of someunknownPlace."""
token_ids = tokenizer.encode(prompt, allowed_special = {"<|endoftext|>"})
print(token_ids)

# print(tokenizer.decode(token_ids=token_ids))
print(tokenizer.decode(token_ids))

token_ids = tokenizer.encode("Akwirw ier", allowed_special = {"<|endoftext|>"})
print(token_ids)
print(tokenizer.decode(token_ids))

with open("the-verdict.txt", "r") as f:
        raw_text = f.read()
dataloader = create_dataloader(raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)
data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)