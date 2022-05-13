import flwr as fl
from datetime import datetime


num_clients = 1
strategy = fl.server.strategy.FedAvg(
  fraction_fit = 0.5,
  fraction_eval = 0.5,
  min_available_clients = num_clients,
  min_fit_clients = num_clients,
  min_eval_clients = num_clients
)

# Get time
start_time = datetime.now()
# Start Flower server
fl.server.start_server(
  "[::]:8080",
  config={"num_rounds": 1},
  strategy=strategy
)
print(f"\nTotal time consumed: {datetime.now() - start_time}")