import pandas as pd
import sys



def load_csv(instrument_to_proccess):
    base_path = "/root/aiRoot/0-AI/AI/recurrentNeuralNetwork/data/"
    file_path = base_path + f"{instrument_to_proccess}_raw.csv"
    try:
        data = pd.read_csv(file_path)
        print("Loaded data")
        return data
    except:
        print("Could not load data")
        sys.exit(0)


def normalize_data(df, columns):
    print("Normalizing data:")
    for col in columns:
        print(f"     Normalizing -> {col}")
        min_val = df[col].min()
        max_val = df[col].max()
        df[col] = (df[col] - min_val) / (max_val - min_val)
    print("Normalization complete:")
    print(df.head(10))
    return df




if __name__ == "__main__":
#------------------------------------------------
    instrument_to_proccess = "NFT"
#------------------------------------------------  
    data = load_csv(instrument_to_proccess)
    print(data.head())
    print("- - - Ready to proccess - - -")
    data = normalize_data(data, ['open', 'high', 'low', 'close'])



