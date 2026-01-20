import pandas as pd

data = pd.read_csv("data/NBA_Multi_Player_Training_Data.csv")
print(data.shape)
print(data.columns)
data.columns.tolist()