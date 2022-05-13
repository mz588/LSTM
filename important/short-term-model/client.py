from collections import OrderedDict
from pickletools import optimize
from posixpath import abspath
from unittest import TestLoader
import warnings

import flwr as fl
import pandas as pd
import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from datetime import datetime 
import numpy as np
from pytorchtools import EarlyStopping
import argparse

# #############################################################################
# Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Define hyper-parameters here
# number of features = 5: Month, Hour, power, steam, chilled_water
# These are the parameters for -min_max.pt
hyper_parameters = {"num_features":5, "num_layers":1, "num_hidden_units":256, "batch_size":256, "epoch":500, "lr":1e-4, "seq_len":4, "step_len":4, "patience":50, "delta":0.001}

# These are the parameters for -2_layer.pt (no better than 1 layer)
# hyper_parameters = {"num_features":5, "num_layers":2, "num_hidden_units":256, "batch_size":256, "epoch":500, "lr":1e-4, "seq_len":4, "step_len":4, "patience":50, "delta":0.001}


class SequenceDataset(Dataset):
    def __init__(self, dataframe, target, features, sequence_length=6):
        self.features = features
        self.target = target
        self.sequence_length = sequence_length
        self.y = torch.tensor(dataframe[self.target].values).float()
        self.X = torch.tensor(dataframe[self.features].values).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i): 
        if i >= self.sequence_length - 1:
            i_start = i - self.sequence_length + 1
            x = self.X[i_start:(i + 1), :]
            y = self.y[i_start:(i+1)].reshape(-1)
        else:
            padding_x = self.X[0].repeat(self.sequence_length-i-1, 1)
            x = self.X[0:(i+1),:]
            x = torch.cat((padding_x,x),0)
            padding_y = self.y[0].repeat(self.sequence_length - i - 1)
            y = self.y[0:(i+1)].reshape(-1)
            y = torch.cat((padding_y, y))

        return x, y


class RegressionLSTM(nn.Module):
    def __init__(self, num_features, hidden_units, out_features):
        super().__init__()
        self.num_features = num_features  # this is the number of features
        self.hidden_units = hidden_units
        self.num_layers = hyper_parameters["num_layers"] # Defines the number of connected LSTM 

        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers=self.num_layers
        )

        self.linear = nn.Linear(in_features=self.hidden_units, out_features=out_features)

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_().to(DEVICE)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_().to(DEVICE)
        
        _, (hn, _) = self.lstm(x, (h0, c0))
        out = self.linear(hn[0]).flatten()  # First dim of Hn is num_layers, which is set to 1 above.

        return out
        
def train_model(data_loader, model, loss_function, optimizer, step_len):
  num_batches = len(data_loader)
  total_loss = 0
  model.train()
  for X, y in data_loader:
    
    X, y = X.to(DEVICE), y.to(DEVICE)
    y = y.reshape(-1)
    output = model(X)
    loss = loss_function(output, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    total_loss += loss.detach().cpu().item()

  avg_loss = total_loss / num_batches
  return avg_loss

def test_model(data_loader, model, loss_function, step_len):
    num_batches = len(data_loader)
    total_loss = 0
    model.eval()
    with torch.no_grad():
      for X, y in data_loader:
        X = X.to(DEVICE)
        y = y.to(DEVICE)
        y = y.reshape(-1)
        output = model(X)
        total_loss += loss_function(output, y).detach().cpu().item()
    if(num_batches == 0):
      print(f"Number of bataches: {num_batches}\n")
    avg_loss = total_loss / num_batches
    return avg_loss, 1 - avg_loss

def predict_model(data_loader, model):
    """Just like `test_loop` function but keep track of the outputs instead of the loss
    function.
    """
    output = torch.tensor([])
    model.eval()
    with torch.no_grad():
        for X, _ in data_loader:
          X = X.to(DEVICE)
          pred = model(X)
          output = torch.cat((output, pred.detach().cpu()), 0)
    
    return output

def train(model, trainLoader, testLoader, loss_function, optimizer, epoches):
  start_time = datetime.now()
  train_losses, test_losses = [], []
  step_len = hyper_parameters["step_len"]
  # early_stopping = EarlyStopping(patience=hyper_parameters["patience"], verbose=True, delta=hyper_parameters["delta"])
  for ix_epoch in range(epoches):
    print(f"Epoch {ix_epoch}\n---------")
    train_loss = train_model(trainLoader, model, loss_function, optimizer=optimizer, step_len=step_len)
    train_losses.append(train_loss)
    print(f"Train loss: {train_loss}")

    test_loss, accuracy = test_model(testLoader, model, loss_function, step_len)
    test_losses.append(test_loss)
    print(f"Test loss: {test_loss}")

    # # Here is early stopping
    # early_stopping(test_loss, model)
    # if early_stopping.early_stop:
    #   print("Early Stopped")
    #   break
  print(f"\nTime consumed: {datetime.now() - start_time}")

def predict(testLoader, testDataset, model, loss_function, step_len):
  label, pred = [], []
  predict_result = predict_model(testLoader, model)
  loss = 0
  combined = pd.DataFrame()
  for i in range(len(testDataset)):
    if i < step_len:
      continue
    x, y = testDataset[i]
    label.append(y.reshape(-1))
    pred.append(predict_result[i * step_len:(i+1) * step_len].reshape(-1))
    loss += loss_function(label[i-step_len], pred[i-step_len])
  combined["Predict"] = pred
  combined["Label"] = label
  loss /= len(testDataset)
  return combined, loss

def load_data(file_name, test_start,test_end, sequence_len, step_len):
  trainloader, testloader, trainDataset, testDataset = None, None, None, None
  abspath = os.path.abspath(file_name).split("/")
  abspath = "/".join(abspath[:-1])
  os.chdir(abspath)
  print(os.getcwd())
  df = pd.read_csv(file_name, index_col="Time")
  df = df.rename(index= dict(zip(df.index, [datetime.strptime(index, '%m/%d/%Y %H:%M:%S') for index in df.index])))
  # start = "2020-01-01"
  # end = "2021-12-31"
  # df = df[start:end]
  
  # Generate label
  # target_col = file_name.replace(".csv","")
  target_col = "Electric"
  target = f"{target_col}_lead{step_len}"
  df[target] = df[target_col].shift(-step_len)
  df = df.iloc[:-step_len]

  # Generate train adn test dataframe
  # test_start = "2022-04-01"
  df_train = df[:test_start]
  df_test = df[test_start:test_end]
  print(f"Train size:{df_train.shape}, test size: {df_test.shape}")

  # min-max normalization
  target_min, target_max = df[target_col].min(), df[target_col].max()
  for c in df.columns:
    # if c != target and c != target_col:
    if c in ["Month", "Hour"]:
      continue
    df_train[c] = (df_train[c] - df_train[c].min()) / (df_train[c].max() - df_train[c].min())
    df_test[c] = (df_test[c] - df_test[c].min()) / (df_test[c].max() - df_test[c].min())

  # Finally generate dataset and loader
  batch_size = hyper_parameters['batch_size']
  features = list(df.columns.difference([target]))
  trainDataset = SequenceDataset(df_train, target, features, sequence_len)
  testDataset = SequenceDataset(df_test, target, features, sequence_len)

  trainloader = DataLoader(trainDataset, batch_size=batch_size, shuffle=False)
  testloader = DataLoader(testDataset, batch_size=batch_size, shuffle=False)
  
  return trainloader, testloader, trainDataset, testDataset

# #############################################################################
# Federating the pipeline with Flower
# #############################################################################

num_features = hyper_parameters["num_features"]
hidden_units = hyper_parameters["num_hidden_units"]
out_features = hyper_parameters['step_len']
learning_rate = hyper_parameters["lr"]
net = None
optimizer = None
loss_function = nn.L1Loss()


# Define Flower client
class FlowerClient(fl.client.NumPyClient):
  def get_parameters(self):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

  def set_parameters(self, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

  def fit(self, parameters, config):
    epoches = hyper_parameters["epoch"]
    self.set_parameters(parameters)
    global trainloader, testloader
    train(net, trainloader, testloader, loss_function, optimizer, epoches=epoches)
    return self.get_parameters(), len(trainloader.dataset), {}

  def evaluate(self, parameters, config):
    self.set_parameters(parameters)
    global testloader
    loss, accuracy = test_model(testloader, net, loss_function, out_features)
    return float(loss), len(testloader.dataset), {"accuracy": float(accuracy)}

# Start Flower client
if __name__ == '__main__':
  global trainloader, testloader, trainDataset, testDataset
  parser = argparse.ArgumentParser(description='Process commandline arguments',conflict_handler='resolve')
  parser.add_argument('--file_name','-f',type=str, required=True,help="File name")
  parser.add_argument('--cuda_num','-c',type=str,help="cuda_number")
  args = parser.parse_args()
  file_name = args.file_name
  if args.cuda_num:
    DEVICE = torch.device("cuda:"+args.cuda_num if torch.cuda.is_available() else "cpu")
  net = RegressionLSTM(num_features=num_features, hidden_units=hidden_units, out_features=out_features).to(DEVICE)
  optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
  trainloader, testloader, trainDataset, testDataset = load_data(file_name, "2022-04-01", "2022-04-31", hyper_parameters["seq_len"], hyper_parameters["step_len"])

  fl.client.start_numpy_client("[::]:8080", client=FlowerClient())
  print("Done!")
  save_model_name = "trained_model.pt"
  torch.save(net.state_dict(), save_model_name)
  print(f"Trained model save to {os.getcwd()}/{save_model_name}")