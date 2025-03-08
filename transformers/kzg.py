import numpy as np
import random
import re
import openai  # OpenAI API for LLM sentence refinement
from sympy import symbols, interpolate
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from collections import defaultdict

# ----------------------- 🔹 CONFIGURABLE VARIABLES -----------------------
SAMPLE_PERCENT = 0.5       # 30% of data used for polynomial fitting
MARKOV_ORDER = 7           # Higher order for better fluency
SENTENCE_LENGTH = 15       # Length of generated sentences
INTERPOLATION_METHOD = 'quadratic'  # Can be 'linear', 'quadratic', or 'cubic'
NOISE_SCALE = 0.1          # Amount of random noise added
SAVE_PLOT = "generated_plot.png"  # Output path for saved plot
USE_LLM_REFINEMENT = False  # Enable LLM-based sentence enhancement
OPENAI_API_KEY = "your-openai-api-key"  # Set your OpenAI API key

print("🔄 Loading dataset...")

# ----------------------- 🔹 Load & Process Dataset -----------------------
try:
    from data import read_data  # Import dataset from external file
    print("✅ Dataset loaded from 'data.py'.")
except ImportError:
    # Default dataset in case `data.py` is missing
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
sample_embeddings = np.array([data_sequence[i] for i in sample_indices]).reshape(-1, 1)

print(f"🔄 Selecting {int(SAMPLE_PERCENT * 100)}% of data for polynomial interpolation...")
print(f"✅ Selected {sample_size} data points for interpolation.")

# ----------------------- 🔹 Apply PCA for Dimensionality Reduction -----------------------
print("🔄 Applying PCA for dimensionality reduction...")
pca = PCA(n_components=1)
sample_embeddings_1d = pca.fit_transform(sample_embeddings).flatten()
data_embeddings_1d = pca.transform(np.array(data_sequence).reshape(-1, 1)).flatten()
print("✅ PCA applied successfully.")

# ----------------------- 🔹 Interpolation Selection -----------------------
print(f"🔄 Performing {INTERPOLATION_METHOD} interpolation...")
interp_func = interp1d(sample_indices, sample_embeddings_1d, kind=INTERPOLATION_METHOD, fill_value="extrapolate")

# Generate New Data Points
new_indices = sorted(set(range(len(data_sequence))) - set(sample_indices))
synthetic_embeddings_1d = interp_func(new_indices)
print(f"✅ Interpolation complete. Generated {len(new_indices)} synthetic data points.")

# ----------------------- 🔹 Add Configurable Noise -----------------------
print(f"🔄 Adding noise (scale: {NOISE_SCALE}) to synthetic data...")
noise = np.random.normal(scale=NOISE_SCALE, size=len(synthetic_embeddings_1d))
synthetic_embeddings_1d += noise
print("✅ Noise added successfully.")

# ----------------------- 🔹 Map Synthetic Data Back to Words -----------------------
print("🔄 Mapping synthetic data back to real words using nearest neighbors...")
nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(data_embeddings_1d.reshape(-1, 1))
_, indices = nbrs.kneighbors(synthetic_embeddings_1d.reshape(-1, 1))
synthetic_words = [index_to_word[data_sequence[i[0]]] for i in indices]
print("✅ Mapping complete.")

# ----------------------- 🔹 Markov Chain for Sentence Generation -----------------------
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
synthetic_text_markov = generate_sentence(markov_chain, START_WORDS, SENTENCE_LENGTH)

print("✅ Markov sentence generation complete.")

# ----------------------- 🔹 LLM-Based Sentence Refinement -----------------------
def refine_with_llm(text):
    print("🔄 Sending sentence to LLM for refinement...")
    openai.api_key = OPENAI_API_KEY
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert in Shakespearean English. Refine the given sentence while preserving its poetic structure."},
            {"role": "user", "content": text}
        ]
    )
    
    refined_text = response["choices"][0]["message"]["content"]
    print("✅ LLM refinement complete.")
    return refined_text

if USE_LLM_REFINEMENT:
    synthetic_text_final = refine_with_llm(synthetic_text_markov)
else:
    synthetic_text_final = synthetic_text_markov

# ----------------------- 🔹 Display Final Generated Synthetic Text -----------------------
print("\n📜 Generated Synthetic Text (LLM-Refined):")
print(synthetic_text_final)

# ----------------------- 🔹 Visualization Plot -----------------------
print("🔄 Creating visualization plot...")
plt.figure(figsize=(8, 4))
plt.plot(sample_indices, sample_embeddings_1d, 'bo', label=f'Sampled Data ({int(SAMPLE_PERCENT * 100)}%)')
plt.plot(new_indices, synthetic_embeddings_1d, 'rx', label=f'Generated Data ({int((1 - SAMPLE_PERCENT) * 100)}%) with Noise')
plt.legend()
plt.title("Interpolation of Word Embeddings (Dynamically Generated)")
plt.xlabel("Word Position")
plt.ylabel("1D Projected Embedding Value")
plt.savefig(SAVE_PLOT)
print(f"✅ Plot saved successfully as '{SAVE_PLOT}'.")
