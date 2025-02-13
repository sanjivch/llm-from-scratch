import torch


input_ids = torch.tensor([2, 3, 5, 1])

class Embedding:

    def __init__(self, vocab_size: int, output_dim:int):
        torch.manual_seed(123)
        self.vocab_size = vocab_size
        self.output_dim = output_dim

        self.token_embedding = torch.nn.Embedding(self.vocab_size, self.output_dim)

# vocab_size = 50257
# output_dim = 256

# token_embedding = torch.nn.Embedding(vocab_size, output_dim)
# position_embedding = ""
# print(token_embedding.weight)
# print(token_embedding(torch.tensor([4])))

# print(token_embedding(input_ids))

