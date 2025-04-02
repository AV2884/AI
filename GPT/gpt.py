import torch
import torch.nn as nn
from torch.nn import functional as F


with open('data.txt', 'r', encoding='utf-8') as f:
    text = f.read()
print(f"Length of dataset: {len(text)} characters")


# STEP 2 : Build vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)
# print(f"Vocabulary: {chars}")
# print(f"Vocab size: {vocab_size}")

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

#Step 3: Encode Entire Dataset
data = torch.tensor(encode(text), dtype=torch.long)
print(data)
print(data.shape)