import pandas as pd
import os

os.makedirs("../data/processed", exist_ok=True)

df = pd.read_parquet("../data/raw/ES_1h.parquet")

#  index → column 
df = df.reset_index()

#  datetime 
df = df.rename(columns={"ts_event": "Datetime"})
df["Datetime"] = pd.to_datetime(df["Datetime"])

#  remove spread instruments
df = df[~df["symbol"].str.contains("-")]

#  rename columns 
df = df.rename(columns={
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "volume": "Volume"
})

#  select most liquid contract per timestamp  
df = df.sort_values(["Datetime", "Volume"], ascending=[True, False])
df = df.drop_duplicates(subset=["Datetime"], keep="first")

# sanity filter 
df = df[
    (df["Low"] > 1000) &
    (df["High"] < 10000)
]

#  sort 
df = df.sort_values("Datetime").reset_index(drop=True)

#  keep only needed 
df = df[["Datetime", "Open", "High", "Low", "Close", "Volume"]]
df.to_parquet("../data/processed/ES_1h.parquet")

print(" Clean ES futures hourly data ready")