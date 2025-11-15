import pandas as pd

spot = pd.read_csv("clean_spot_prices_2023_ap-northeast-1.csv")

specs = pd.read_csv("instance_specs.csv")

spot_summary = (
    spot
    .groupby("instance_type")["spot_price"]
    .mean()
    .reset_index()
)
spot_summary.rename(columns={"spot_price": "avg_spot_price"}, inplace=True)

merged = specs.merge(spot_summary, on="instance_type", how="left")

merged.to_csv("merged_instance_prices.csv", index=False)

print("Done! Saved merged_instance_prices.csv")
print(merged.head())
