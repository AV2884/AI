import numpy as np
import time
from itertools import permutations

# Generate a smaller set of currencies for easier simulation and understanding
currencies = ["USDT", "ETH", "BTC"]  # Example with 3 currencies
exchange_rates = {
    ("USDT", "ETH"): 0.0002,  # USDT to ETH
    ("ETH", "BTC"): 0.06,     # ETH to BTC
    ("BTC", "USDT"): 16000,   # BTC to USDT
    ("USDT", "BTC"): 0.0000625,  # Reverse exchange rates for illustration
    ("ETH", "USDT"): 5000,    # Reverse rate for ETH to USDT
    ("BTC", "ETH"): 16.67     # Reverse rate for BTC to ETH
}

# Convert exchange rates into a matrix for easy lookup
currency_index = {currency: idx for idx, currency in enumerate(currencies)}
num_currencies = len(currencies)
rate_matrix = np.zeros((num_currencies, num_currencies))

# Fill the rate matrix with exchange rates
for (currency_a, currency_b), rate in exchange_rates.items():
    i = currency_index[currency_a]
    j = currency_index[currency_b]
    rate_matrix[i, j] = rate

# Function to calculate potential profits for a given path
def calculate_profit(path):
    balance = 100  # Starting balance in USDT
    for i in range(len(path) - 1):
        from_currency = path[i]
        to_currency = path[i + 1]
        rate = rate_matrix[currency_index[from_currency], currency_index[to_currency]]
        balance *= rate
    return balance - 100  # Profit or loss compared to initial balance

# Generate all possible 3-leg paths
def generate_paths():
    paths = []
    for perm in permutations(currencies, 3):
        paths.append(list(perm) + [perm[0]])  # Loop back to the starting currency
    return paths

# Calculate profits for all paths and find the best one
paths = generate_paths()
profits = []
start_time = time.time()

for path in paths:
    profit = calculate_profit(path)
    profits.append((path, profit))

end_time = time.time()

# Find the best path
best_path, best_profit = max(profits, key=lambda x: x[1])

# Print all paths with their profits
print("All Possible Paths with Profits/Losses:")
for path, profit in profits:
    print(f"{' -> '.join(path)}: {'Profit' if profit > 0 else 'Loss'} {abs(profit):.2f} USD")

# Print the best path
print("\nBest Arbitrage Path:")
print(f"Opening Balance: 100 USD")
print(f"{' -> '.join(best_path)}: {'Profit' if best_profit > 0 else 'Loss'} {abs(best_profit):.2f} USD")
print(f"Time taken using CPU: {end_time - start_time:.2f} seconds")
