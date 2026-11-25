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

data = np.array(encode(text))
# print(data.shape)
# print(data[:5])
# print(decode(data[:13]))

N = np.zeros((len(char),len(char)),dtype=int)

for i in range(len(data)-1):
    N[data[i], data[i+1]] += 1

sum_of_rows = N.sum(axis=1,keepdims=True)
P = N / sum_of_rows
# print(P[stoi["q"], stoi["u"]])
# print(P[stoi["t"], stoi["h"]])
# print(P.sum(axis=1)[:5])

# rng = np.random.default_rng(1337)
# out = []
# current = 0
# for i in range(300):
#     current = rng.choice(65, p=P[current])
#     print(current)
#     out.append(current)
# print(decode(out))

# print(-np.log(P[stoi["i"], stoi["r"]]))
# print(-np.log(P[stoi["r"], stoi["s"]]))
# print(-np.log(P[stoi["s"], stoi["t"]]))

# print(1 / P[stoi["i"], stoi["r"]])
# print(np.exp(2.8252734093025484))

total = 0

for i in range(len(data)-1):
    total += -np.log(P[data[i], data[i+1]])

total = total / (len(data) - 1)
print(total)
print(-np.log(1 / 65))