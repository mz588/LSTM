import pandas as pd
import os
import torch
from torch import nn
import matplotlib.pyplot as plt
import argparse

from client import load_data, predict, RegressionLSTM, DEVICE

hyper_parameters = {"num_features":3, "num_layers":1, "num_hidden_units":100, "batch_size":128, "epoch":200, "lr":5e-4, "seq_len":24, "step_len":24}

def plot(result):
  folder_name = "img_result"
  if not os.path.exists(folder_name):
    os.makedir(folder_name)
  os.chdir(folder_name)
  predicts, label = result.columns
  for index in range(len(result[label])):
    plt.figure()
    plt.plot(result[label][index].reshape(-1), '.-', label="Original")
    plt.plot(result[predicts][index].reshape(-1), '.-', label="Predicts")
    plt.legend()
    plt.savefig(f"output_{index}_{index+24}.jpg")

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Process commandline arguments',conflict_handler='resolve')
  parser.add_argument('--file_name','-f',type=str, required=True,help="File name")
  parser.add_argument('--path','-p',type=str, required=True,help="Path of the file")
  parser.add_argument('--model', '-m', type=str, required=True,help="Model name with complete path")
  args = parser.parse_args()
  file_name = args.file_name
  file_path = args.path
  model_file = args.model

  features = hyper_parameters["num_features"]
  num_hiddens = hyper_parameters["num_hidden_units"]
  step_len = hyper_parameters["step_len"]
  model = RegressionLSTM(num_features=features, hidden_units=num_hiddens, out_features=step_len).to(DEVICE)
  model.load_state_dict(torch.load(model_file,map_location=DEVICE))
  
  trainloader, testloader, trainDataset, testDataset = load_data(file_path, file_name)
  combined, loss = predict(testloader, testDataset, model, loss_function=nn.L1Loss())
  print(f"\n\nFile: {file_path+file_name}, loss: {loss}")
  plot(combined)