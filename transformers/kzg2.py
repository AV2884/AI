import numpy as np
import random
import re
from sympy import symbols, interpolate
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from collections import defaultdict
from py_ecc.bn128 import G1, G2, Z1, add, multiply, pairing
from tqdm import tqdm

# ----------------------- 🔹 CONFIGURABLE VARIABLES -----------------------
SAMPLE_PERCENT = 0.3       # 30% of data used for polynomial fitting
MARKOV_ORDER = 4           # Higher order for better fluency
SENTENCE_LENGTH = 15       # Length of generated sentences
NOISE_SCALE = 0.1          # Amount of random noise added
SAVE_PLOT = "generated_plot.png"  # Output path for saved plot

print("🔄 Loading dataset...")

# ----------------------- 🔹 Load & Process Dataset -----------------------
try:
    from data import read_data  # Import dataset from external file
    print("✅ Dataset loaded from 'data.py'.")
except ImportError:
    read_data = """
    ACT I
    My lord, what dost thou think of the matter?
    'Tis but a fleeting dream, yet truth doth lie within.
    Speak, O noble one, for time hast given thee wisdom.
    Love and honor do dance upon the wind of fate.
    """
    print("⚠️ 'data.py' not found. Using default Shakespeare-style dataset.")

print("🔄 Tokenizing dataset...")

# Tokenization & Word Indexing
words = read_data.replace("\n", " ").split()
word_to_index = {word: i for i, word in enumerate(set(words))}
index_to_word = {i: word for word, i in word_to_index.items()}
data_sequence = [word_to_index[word] for word in words]

print(f"✅ Tokenization complete. Total words: {len(words)}")

# 🔹 Set START_WORDS to the first 2 words of the dataset
START_WORDS = words[:2] if len(words) >= 2 else ["ACT", "I"]
print(f"🔹 Automatically set START_WORDS to: {START_WORDS}")

# ----------------------- 🔹 Select Sample Data -----------------------
sample_size = int(len(data_sequence) * SAMPLE_PERCENT)  
sample_indices = sorted(random.sample(range(len(data_sequence)), sample_size))
sample_data = [data_sequence[i] for i in sample_indices]

print(f"🔄 Selected {sample_size} data points for KZG polynomial fitting.")

# ----------------------- 🔹 Step 1: Create KZG Polynomial -----------------------
x = symbols('x')
# polynomial = interpolate(list(zip(sample_indices, sample_data)), x)
polynomial = tqdm(interpolate(list(zip(sample_indices, sample_data)), x), desc="pol_fit")
# polynomial = interpolate(list(zip(sample_indices, tqdm(sample_data, desc="Interpolating"))), x)
# print(f"✅ Polynomial constructed: {polynomial}")

# ----------------------- 🔹 Step 2: Generate KZG Commitment -----------------------
# Trusted Setup for KZG
G = G1  # Generator in the elliptic curve
tau = random.randint(1, 10**9)  # Secret randomness (simulated for testing)

# Compute commitments for each coefficient
coefficients = polynomial.as_coefficients_dict()
# commitment = sum(multiply(G, coeff * tau**i) for i, coeff in coefficients.items())
commitment = sum(
    multiply(G, coeff * tau**i) for i, coeff in tqdm(coefficients.items(), desc="Computing Commitment")
)
print("✅ KZG commitment generated.")

# ----------------------- 🔹 Step 3: Generate New Data & Verify with KZG -----------------------
new_indices = sorted(set(range(len(data_sequence))) - set(sample_indices))
new_values = [polynomial.subs(x, i) for i in new_indices]  # Evaluate the polynomial at new indices

# Convert values to nearest word indices
synthetic_data = [int(val) % len(word_to_index) for val in new_values]
synthetic_words = [index_to_word[i] for i in synthetic_data]

print("✅ New data points generated and mapped to words.")

# ----------------------- 🔹 Step 4: Verify Interpolated Data Using KZG -----------------------
def verify_kzg(x_value, y_value):
    """Verify if the interpolated value (x_value, y_value) is part of the committed polynomial."""
    witness = multiply(G, y_value)  # Compute witness
    check = pairing(witness, multiply(G2, tau)) == pairing(commitment, multiply(G2, x_value * tau))
    return check

print("🔄 Verifying interpolated data points with KZG commitments...")

verified_points = []
for i, y in zip(new_indices, synthetic_data):
    if verify_kzg(i, y):
        verified_points.append(i)

print(f"✅ Verified {len(verified_points)} / {len(new_indices)} interpolated points.")

# ----------------------- 🔹 Step 5: Use Markov Chain for Sentence Generation -----------------------
print(f"🔄 Building Markov chain (order: {MARKOV_ORDER})...")

def build_markov_chain(words, order=MARKOV_ORDER):
    markov_chain = defaultdict(list)
    for i in range(len(words) - order):
        key = tuple(words[i:i+order])
        markov_chain[key].append(words[i+order])
    return markov_chain

def generate_sentence(markov_chain, start_word, length=SENTENCE_LENGTH):
    print(f"🔄 Generating sentence using Markov Chain (length: {length})...")
    sentence = list(start_word)
    for _ in range(length):
        key = tuple(sentence[-MARKOV_ORDER:])
        next_word = random.choice(markov_chain.get(key, words))
        sentence.append(next_word)
    return " ".join(sentence)

markov_chain = build_markov_chain(words, order=MARKOV_ORDER)
synthetic_text = generate_sentence(markov_chain, START_WORDS, SENTENCE_LENGTH)

print("✅ Markov sentence generation complete.")

# ----------------------- 🔹 Display Final Generated Synthetic Text -----------------------
print("\n📜 Generated Synthetic Text:")
print(synthetic_text)

# ----------------------- 🔹 Visualization Plot -----------------------
