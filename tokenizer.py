import re

class NaiveTokenizer:

    def __init__(self, vocab: dict):
        self.token2id = vocab
        self.id2token = {token_id:token for token, token_id in vocab.items()}

    def encode(self, text: str) -> list:
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [i.strip() for i in preprocessed if i.strip()]
        preprocessed = [i if i in self.token2id else "<|unk|>" for i in preprocessed]

        return [self.token2id[token] for token in preprocessed]
    
    def decode(self, token_ids: list) -> str:
        print([token_id for token_id in token_ids])
        text = " ".join([self.id2token[token_id] for token_id in token_ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        
        return text
    


        