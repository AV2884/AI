import numpy as np

# 1. load the book
with open("data/input.txt", encoding="utf-8") as f:
    text = f.read()
print(f"book: {len(text)} characters")

# 2. tokenizer: every character gets a number, its position in chars
chars = sorted(set(text))
stoi = {}
itos = {}

for i in range(len(chars)):
    stoi[chars[i]] = i

for i in range(len(chars)):
    itos[i] = chars[i]

def encode(msg):
    encoded = []
    for c in msg:
        encoded.append(stoi[c])
    return encoded

def decode(ids):
    decoded = ""
    for i in ids:
        decoded += itos[i]
    return decoded

print(f"vocabulary: {len(chars)} characters")
print(f"hello -> {encode('hello')}")
print(f"round trip ok: {decode(encode(text)) == text}")

# 3. the whole book as an array of numbers
data = np.array(encode(text))
print(f"data: {data.shape} starts {data[:5]}")

# 4. count the pairs: N[first, next] = how many times next followed first
N = np.zeros((len(chars), len(chars)), dtype=int)

for i in range(len(data) - 1):
    N[data[i], data[i + 1]] += 1

print(f"pairs counted: {N.sum()}, th: {N[stoi['t'], stoi['h']]}")

# 5. counts to chances: divide every row by its own total
sum_of_rows = N.sum(axis=1, keepdims=True)
P = N / sum_of_rows
print(f"chance of h after t: {P[stoi['t'], stoi['h']]:.3f}")

# 6. the model writes: stand on a character, roll its row, move there, repeat
rng = np.random.default_rng(1337)
out = []
current = 0
for i in range(300):
    current = rng.choice(len(chars), p=P[current])
    out.append(current)

print("--- the model writes ---")
print(decode(out))

# 7. score the model: average of -log(chance it gave to what really came next)
total = 0

for i in range(len(data) - 1):
    total += -np.log(P[data[i], data[i + 1]])

loss = total / (len(data) - 1)
print("--- score ---")
print(f"model {loss:.4f}, know-nothing {-np.log(1 / len(chars)):.4f}")