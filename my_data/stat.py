import pandas as pd

df = pd.read_csv("./csv/filtered_all.csv")

print("Min", df["len"].min())
print("Max", df["len"].max())
print("Total", len(df))
filter_df = df[(df["len"] > 150000) | (df["len"] < 22050)]
print("> 6s:", len(filter_df))

# filter_df.to_csv("./csv/filtered_all.csv", index=False)
