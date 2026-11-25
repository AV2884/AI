import numpy as np

with open("data/input.txt", encoding="utf-8") as f:
    text = f.read()

char = list(sorted(set(text)))
stoi = {}
itos = {}

for i in range(len(char)):
    stoi[char[i]] = i

for i in range(len(char)):
    itos[i] = char[i] 

def encode(msg):
    encoded = []
    for char in msg:
        asign = stoi[char]
        encoded.append(asign)
    return encoded


def decode(msg):
    decoded = ""

    for i in msg:
        decoded += itos[i]

    return decoded

enc1 = encode("hello")
print(enc1)
dec1 = decode(enc1)
print(dec1)
print(decode(encode(text)) == text)

data = np.array(encode(text))
print(data.shape)
print(data[:5])
print(decode(data[:13]))