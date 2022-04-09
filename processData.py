import pandas as pd
from datetime import datetime
import numpy as np

df = pd.read_csv("NYISO.csv")

df.drop(['PTID', 'Bilateral Load', 'Price Cap Load', 'Virtual Load', 'Virtual Supply'], axis = 1, inplace=True)
df = df.set_index("Time Stamp", drop=True).sort_index()
df = df.rename(index= dict(zip(df.index, [datetime.strptime(index, '%m/%d/%Y %H:%M:%S') for index in df.index])))
df.sort_values("Time Stamp", inplace=True, ascending=True)

df_cleaned = pd.DataFrame(index=df.index.unique())
names = list(df["Name"].unique())
names.remove("N.Y.C._LONGIL")

df_all = {}
for i in range(len(names)):

  temp = pd.DataFrame({names[i]:df[df["Name"] == names[i]]["Energy Bid Load"].astype(np.float32)})
  temp["Hour"] = temp.index.hour
  temp["Hour"] = temp["Hour"].astype('int32')
  temp["Month"] = temp.index.month
  temp["Month"] = temp["Month"].astype('int32')
  temp = temp.rename(index = dict(zip(temp.index, [index.strftime('%m/%d/%Y %H:%M:%S') for index in temp.index])))
  df_all[names[i]] = temp
  file_name = "./data/"+names[i]+".csv"
  print(file_name)
  df_all[names[i]].to_csv(file_name)
