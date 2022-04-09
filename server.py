import flwr as fl
num_clients = 5
strategy = fl.server.strategy.FedAvg(
  fraction_fit = 0.5,
  fraction_eval = 0.5,
  min_available_clients = num_clients,
  min_fit_clients = num_clients,
  min_eval_clients = num_clients
)

# Start Flower server
fl.server.start_server(
  "[::]:8080",
  config={"num_rounds": 6},
  strategy=strategy
)