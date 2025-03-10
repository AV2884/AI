import multiprocessing
import time

def intensive_task(n):
    primes = []
    for num in range(2, n):
        if all(num % i != 0 for i in range(2, int(num ** 0.5) + 1)):
            primes.append(num)
    return primes

if __name__ == "__main__":
    num_cores = multiprocessing.cpu_count()
    n = 5_00_00_000  

    print(f"Using {num_cores} cores for computation...")
    start_time = time.time()

    pool = multiprocessing.Pool(num_cores)
    chunk_size = n // num_cores

    # Distribute workload evenly across all CPU cores
    results = pool.map(intensive_task, [chunk_size] * num_cores)

    pool.close()
    pool.join()

    end_time = time.time()
    print(f"\nComputation finished in {end_time - start_time:.2f} seconds.")
