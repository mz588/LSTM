# LSTM for predicting power consumption at Cornell University
## This code is based on [Flower](https://flower.dev/)

## Dataset
The dataset used in the project are those collected and maintained by Cornell University. Here is the [website](https://portal.emcs.cornell.edu/d/2/dashboard-list?orgId=2). The lowest sampling rate is 15 minutes per point. And it also provides the aggregated data for every hour and everyday. Although this website supports downloading the csv file of the dataset, there are some constraints. First, the maximum size of the file cannot exceed around 2.5 MB. So, it is impossible to select and download the dataset for the entire previous year all at once. Second, because sensors could mulfunction from time to time, please do double-check the download csv file and remember to do cleaning. Finally, because only a few features are sharded among most of the buildings, in this project, we use the chilled water, electricity, and steam in the past as features to perform training and inference.

The data from March 2021 to April 2022 for Upson Hall, Carpenter Hall, and Sage Hall have been cleaned and stored in UpsonHall.csv, CarpenterHall.csv, and SageHall.csv for sampling rate of 5min and 1h. You can find them inside the `Cornell_data_5min` and `Cornell_data_1h` folder.

Moreover, the script for cleaning the data is also included in preProcess.py.

## Description on the code structure
As the title implies, we use LSTM in this project, which has been included in the library of Pytorch. 

There are two models in this git repo. The [short-term prediction](./important/short-term-model/) and the [long-term prediction](./important/long-term-model/). 

Each of these two folders contain the client.py, server.py, checkResult.py, and some trained models. When you want to run the code and train your own model, here are the detailed procedures on how to initiate training on short-term model:
```sh
$ cd ./important/short-term-model/
$ python server.py
$ python client.py -f ../Cornell_data_5min/SageHall.csv -c 0
$ python client.py -f ../Cornell_data_5min/UpsonHall.csv -c 1
$ python client.py -f ../Cornell_data_5min/CarpenterHall.csv -c 2
```
If you happen to be rich and possess multiple GPUs (just like me), you can do training on separated GPUs simultaneously. 

To check the result and generate image results:
```sh
$ cd ./important/short-term-model/
$ python checkResult.py -f ../Cornell_data_5min/SageHall.csv -m short_trained_model-min_max.pt
```

These are the examples for the short-term prediction, but they can also be applied to the long-term. One thing to notice is that the folder name should be changed from `../Cornell_data_5min/` to `../Cornell_data_1h/`. Also, the command for running checkResult.py need to add `-long` tag. Otherwise, the image results for short and long term will be written into one folder and thus mixed together.

If you want to change the number of clients, see server.py and change `min_available_clients`. 

### Short-term prediction
The target of short-term prediction is to predict the power load in the next hour. As the data used to perform the training on short-term prediction is sampled every 15 minutes, the LSTM model is connected to a fully-connected layer with 4 outputs. All the hyper-parameters are included in hyper_parameters dict in the client.py and checkResult.py file. Be sure to change them accordinly. 

### Long-term prediction
The target of short-term prediction is to predict the power load in the next 24h. As the data used to perform the training on long-term prediction is the aggregated data for 1 hour, the LSTM model is connected to a fully-connected layer with 24 outputs.

## Some future work
1. Try to use weather data as features.
2. Try more rounds of update on weights in server model and less epoches on clients.
3. Try other parameters for weighted averaging.
