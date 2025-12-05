import pandas as pd
df = pd.read_csv("dataset/sitting.csv")
print(df.columns.tolist())
print(len(df.columns))
