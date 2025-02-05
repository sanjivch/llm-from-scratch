import urllib.request
import json
import re
from tokenizer import NaiveTokenizer

def download_data(url: str, file_path: str):
    urllib.request.urlretrieve(url, file_path)
    return


def create_vocab(file_name: str) -> dict:

    # Read text from the file
    with open("the-verdict.txt", "r") as f:
        raw_text = f.read()

    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    preprocessed = [i.strip() for i in preprocessed if i.strip()]

    tokens = sorted(set(preprocessed))
    return {token:token_id for token_id, token in enumerate(tokens)}

    





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

tokenizer = NaiveTokenizer(vocab)
prompt =  """"It's the last he painted, you know," 
       Mrs. Gisburn said with pardonable pride."""
token_ids = tokenizer.encode(prompt)
print(token_ids)

print(tokenizer.decode(token_ids=token_ids))
