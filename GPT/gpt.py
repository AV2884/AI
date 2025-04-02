import torch
import torch.nn as nn
from torch.nn import functional as F
#PARAMS:--------------------------------------------------
DATA_SPLIT_RATIO = 0.9
BLOCK_SIZE = 8  # context length: how many characters to use to predict the next one
BATCH_SIZE = 4  # how many sequences to train on in parallel
#------------------------------------------------------------

with open('data.txt', 'r', encoding='utf-8') as f:
    text = f.read()
print(f"Length of dataset: {len(text)} characters")


# STEP 2 : Build vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Vocabulary: {chars}")
print(f"Vocab size: {vocab_size}")

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

#Step 3: Encode Entire Dataset
data = torch.tensor(encode(text), dtype=torch.long)
print(data)
print(data.shape)

#Step 4: Train/Validation Split
n = int(DATA_SPLIT_RATIO * len(data))
train_data = data[:n]
val_data = data[n:]

print(f"Train length: {len(train_data)}")
print(f"Val length: {len(val_data)}")

#Step 5: Create Training Batches
def get_batch(split):
    data_split = train_data if split == "train" else val_data
    ix = torch.randint(len(data_split) - BLOCK_SIZE, (BATCH_SIZE,)) #This generates batch_size number of random starting positions in the data. making sure we dont exeed the 8 window block from data
    x = torch.stack([data_split[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data_split[i+1:i+BLOCK_SIZE+1] for i in ix])
    return x, y


#Step 6: Build the Tiny Bigram Language Model

# This is your embedding table — same as before
token_embedding_table = torch.nn.Embedding(vocab_size, vocab_size)

def forward(idx, targets=None):
    # Get logits from embeddings
    logits = token_embedding_table(idx)  # shape (B, T, C)

    # If targets provided, compute loss
    if targets is not None:
        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        targets = targets.view(B*T)
        loss = F.cross_entropy(logits, targets)
    else:
        loss = None

    return logits, loss

class DummyModel(torch.nn.Module):
    def __init__(self, embedding):
        super().__init__()
        self.embedding = embedding

dummy_model = DummyModel(token_embedding_table)
optimizer = torch.optim.AdamW(dummy_model.parameters(), lr=1e-3)

max_iters = 30000
eval_interval = 10

for step in range(max_iters):
    
    # every eval_interval steps, print loss
    if step % eval_interval == 0:
        xb, yb = get_batch('val')
        _, loss = forward(xb, yb)
        print(f"Step {step}: val loss {loss.item():.4f}")

    # get batch
    xb, yb = get_batch('train')

    # forward pass
    logits, loss = forward(xb, yb)

    # backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
