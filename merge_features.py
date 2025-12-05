import pandas as pd
import glob

files = glob.glob("dataset/features_*.csv")

dfs = [pd.read_csv(f) for f in files]
df = pd.concat(dfs, ignore_index=True)
df.to_csv("dataset/features_all.csv", index=False)

print("Total samples:", len(df))
print("Saved to dataset/features_all.csv")
