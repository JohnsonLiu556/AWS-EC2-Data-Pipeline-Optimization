import pandas as pd

df = pd.read_csv("data.ap-northeast-1.2023-01-01T00:05:47-08:00.xz",
                 compression="xz",
                 header=None,
                 names=["raw"])

split_df = df["raw"].str.split("\t", expand=True)

split_df.columns = [
    "record_type",
    "spot_price",
    "timestamp",
    "instance_type",
    "product_description",
    "availability_zone"
]

split_df["spot_price"] = pd.to_numeric(split_df["spot_price"], errors="coerce")

split_df.to_csv("clean_spot_prices_2023_ap-northeast-1.csv", index=False)

print("Done! Saved clean_spot_prices_2023_ap-northeast-1.csv")
print(split_df.head())
