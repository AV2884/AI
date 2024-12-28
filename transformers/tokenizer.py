import re

class GPTTokenizer:
    def __init__(self):
        # Initialize vocabulary and special tokens
        self.vocab = {}
        self.inverse_vocab = {}
        self.special_tokens = {"[PAD]": 0, "[UNK]": 1, "[EOS]": 2}
        self.next_id = len(self.special_tokens)

        # Add special tokens to vocab
        for token, id in self.special_tokens.items():
            self.vocab[token] = id
            self.inverse_vocab[id] = token

    def build_vocab(self, corpus):
        """Build vocabulary from a given corpus (list of strings)."""
        word_freq = {}
        for text in corpus:
            tokens = self._tokenize_text(text)
            for token in tokens:
                word_freq[token] = word_freq.get(token, 0) + 1

        # Sort by frequency and assign IDs
        sorted_vocab = sorted(word_freq.keys(), key=lambda x: word_freq[x], reverse=True)
        for token in sorted_vocab:
            if token not in self.vocab:
                self.vocab[token] = self.next_id
                self.inverse_vocab[self.next_id] = token
                self.next_id += 1

    def _tokenize_text(self, text):
        """Tokenize a single string into words and punctuation."""
        return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)

    def tokenize(self, text):
        """Convert text to a list of token IDs."""
        tokens = self._tokenize_text(text)
        return [self.vocab.get(token, self.special_tokens["[UNK]"]) for token in tokens]

    def detokenize(self, token_ids):
        """Convert a list of token IDs back to text."""
        tokens = [self.inverse_vocab.get(id, "[UNK]") for id in token_ids]
        return " ".join(tokens)

# Example Usage
corpus = [
    "Hello world! GPT models are great.",
    "Tokenization is fun and useful.",
]

# Initialize tokenizer and build vocab
tokenizer = GPTTokenizer()
tokenizer.build_vocab(corpus)

# Tokenize a sentence
text = "Hello world! Tokenization is amazing."
token_ids = tokenizer.tokenize(text)
print("Token IDs:", token_ids)

# Detokenize the tokens back to text
decoded_text = tokenizer.detokenize(token_ids)
print("Decoded Text:", decoded_text)
