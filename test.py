import numpy as np
import time

def create_large_data():
    try:
        print("Creating large data structures to consume RAM...")
        data = []
        for _ in range(100):
            large_array = np.random.rand(1000, 1000) 
            data.append(large_array)
            print(f"Allocated chunk: {len(data)}")
            time.sleep(0.5)  
    except MemoryError:
        print("MemoryError! System ran out of RAM.")
    finally:
        input("Press Enter to release memory and end the script...")
        data = None 
if __name__ == "__main__":
    create_large_data()
