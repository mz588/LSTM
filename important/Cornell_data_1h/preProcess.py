from ast import arg
from cmath import nan
from statistics import median
import pandas as pd
from datetime import datetime
import numpy as np

import os
import argparse

def processData(folder_name):
  print(f"{os.getcwd()}, realPath: {os.path.realpath(folder_name)}")
  parent_folder = os.path.abspath(folder_name)
  parent_folder = "/".join(parent_folder.split("/")[:-1])
  print(parent_folder)
  os.chdir(os.path.realpath(folder_name))
  files = os.listdir()
  df = pd.DataFrame()
  for file in files:
    if ".xlsx" in file:
      df = pd.concat([df,pd.read_excel(file)])
  if('slottype' in df.columns): df.drop(['slottype'], axis=1, inplace=True)
  df.rename(columns={"slottime_GMT":"Time","slotavg":"Data"}, inplace=True)
  df.drop_duplicates(inplace=True)

  df = df.set_index("Time", drop=True).sort_index()
  df = df.rename(index= dict(zip(df.index, [datetime.strptime(index, '%Y-%m-%d %H:%M:%S') for index in df.index])))
  features = df.pointTitle.unique()
  for feature in features:
    df[feature] = df[df["pointTitle"] == feature]["Data"]
    df[feature] = df[feature].fillna(method="ffill")

  df.drop(["pointTitle","name","Data"], inplace=True, axis=1)
  df.drop_duplicates(inplace=True)
  df["Month"] = df.index.month.astype('int32')
  df["Hour"] = df.index.hour.astype('int32')
  df = df.rename(index = dict(zip(df.index, [index.strftime('%m/%d/%Y %H:%M:%S') for index in df.index])))
  os.chdir(parent_folder)
  df.to_csv(folder_name+".csv")
  

if __name__ == '__main__':
  global trainloader, testloader, trainDataset, testDataset 
  parser = argparse.ArgumentParser(description='Process commandline arguments',conflict_handler='resolve')
  parser.add_argument('--folder_name','-f',type=str, required=True,help="File name")
  args = parser.parse_args()

  processData(args.folder_name)





