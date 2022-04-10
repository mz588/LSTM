# LSTM for predicting power consumption at Cornell University
## This code is based on [Flower](https://flower.dev/)
## How to run?
First run server:
```sh
$ python3 server.py
```

Then run all clients:
```sh
$ python3 client.py -f CAPITL.csv -p ./data
$ python3 client.py -f N.Y.C..csv -p ./data
$ python3 client.py -f WEST.csv -p ./data
```

If you want to change the number of clients, see [server.py](./server.py) and change `min_available_clients`. 