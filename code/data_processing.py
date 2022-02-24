import pandas as pd

# loading 3 csv files
message_data = pd.read_csv('test_data/34200_4200_NFLX_2017-05-01_24900000_57900000_message_50.csv')
order_data = pd.read_csv('test_data/34200_4200_NFLX_2017-05-01_24900000_57900000_orderbook_50.csv')
index_data = pd.read_csv('test_data/34200_4200_NFLX_2017-05-01_dataset_50.csv',header = 0)

# Concatenate message csv

