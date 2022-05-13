from typing_extensions import Required
import pandas as pd
import os
import torch
from torch import nn
import matplotlib.pyplot as plt
import argparse

from client import predict, RegressionLSTM, load_data

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hyper_parameters = {"num_features":5, "num_layers":1, "num_hidden_units":512, "batch_size":256, "epoch":500, "lr":1e-4, "seq_len":24, "step_len":24, "patience":50, "delta":0.001}

def plot(result, long, file_name):
  if(long): folder_name = "img_result_long_"+file_name
  else: folder_name = "img_result_short_"+file_name
  if not os.path.exists(folder_name):
    os.mkdir(folder_name)
  os.chdir(folder_name)
  predicts, label = result.columns
  for index in range(len(result[label])):
    plt.figure()
    plt.plot(result[label][index].reshape(-1), '.-', label="Original")
    plt.plot(result[predicts][index].reshape(-1), '.-', label="Predicts")
    plt.ylim(0,1)
    plt.legend()
    plt.savefig(f"output_{index}_{index+hyper_parameters['step_len']}.jpg")

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Process commandline arguments',conflict_handler='resolve')
  parser.add_argument('--file_name','-f',type=str, required=True,help="File name")
  parser.add_argument('--model', '-m', type=str, required=True,help="Model name with complete path")
  parser.add_argument("--long", '-l', action='store_true')
  args = parser.parse_args()
  file_name = args.file_name
  model_file = args.model

  features = hyper_parameters["num_features"]
  num_hiddens = hyper_parameters["num_hidden_units"]
  step_len = hyper_parameters["step_len"]
  model = RegressionLSTM(num_features=features, hidden_units=num_hiddens, out_features=step_len, num_layers=hyper_parameters["num_layers"]).to(DEVICE)
  model.load_state_dict(torch.load(model_file,map_location=DEVICE))
  
  trainloader, testloader, trainDataset, testDataset = load_data(file_name, "2022-04-10","2022-04-20", hyper_parameters["seq_len"],hyper_parameters["step_len"])
  combined, loss = predict(testloader, testDataset, model, loss_function=nn.L1Loss(), step_len=step_len)
  print(f"\n\nFile: {file_name}, loss: {loss}")
  plot(combined, args.long, file_name.split("/")[-1].replace(".csv",""))