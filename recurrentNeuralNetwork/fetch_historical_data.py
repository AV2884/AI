import os
import pandas as pd
from dotenv import load_dotenv
from kiteconnect import KiteConnect
from datetime import datetime, timedelta

load_dotenv()

#Global vars:
API_KEY = os.getenv('API_KEY')
API_SECRET = os.getenv('API_SECRET')
ACCESS_TOKEN = os.getenv('ACCESS_TOKEN')
NFT = 256265
BNK = 260105
MID = 288009
FETCH_LIMIT_DAYS = 60
kite = KiteConnect(api_key=API_KEY)
kite.set_access_token(ACCESS_TOKEN)

def describe_data(df):
    print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
    print(df.describe())
    print("-------------------------------------------------------------------")
    print(df.head(10))
    print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")

def fetch_historical_data(instrument_token, from_date, to_date, interval):
    data = kite.historical_data(
        instrument_token=instrument_token,
        from_date=from_date,
        to_date=to_date,
        interval=interval
    )
    return pd.DataFrame(data)


def save_to_csv(df,intrument_token):
    base_path = "/root/aiRoot/0-AI/AI/recurrentNeuralNetwork/data/"
    if intrument_token == 256265:
        index = "NFT"
    elif intrument_token == 260105:
        index = "BNK"
    elif intrument_token == 288009:
        index = "MID"
    file_path = base_path + f"{index}_raw.csv"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=False)
    print(f"Data fetching complete. Saved to {file_path}")


if __name__ == "__main__":
#------------------------------------------------
    DAYS = 730
    INTERVEL = "minute"
    instrument_to_fetch = NFT
#------------------------------------------------

    to_date = datetime.now()
    from_date = to_date - timedelta(days=DAYS)
    print(f"{to_date} --> {from_date}")

    current_start = from_date
    data = pd.DataFrame() 


    while current_start < to_date:

        current_end = min(current_start + timedelta(days=60) ,to_date)
        print(f"Fetching data from {current_start} to {current_end}...")
        chunk_data = fetch_historical_data(
            instrument_token=instrument_to_fetch,
            from_date=current_start,
            to_date=current_end,
            interval=INTERVEL
        )
        data = pd.concat([data, chunk_data], ignore_index=True)
        current_start = current_end

    describe_data(data)
    save_to_csv(data,instrument_to_fetch)



