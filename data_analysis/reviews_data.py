import os
import pandas as pd

#set wd
cwd = "/home/samuel/Work/coffe3/data/"
data_sets = os.listdir(cwd)
data_sets

#read
data_part2 = pd.read_csv(cwd + data_sets[0])
data_part3 = pd.read_csv(cwd + data_sets[1])
data_part1 = pd.read_csv(cwd + data_sets[2])

#total rows
total_rows = data_part1.shape[0] + data_part2.shape[0] + data_part3.shape[0]

list(data_part1)

#describe numeric components
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)

list(data_part1.select_dtypes(include=["floating", "integer"]))
data_part1.select_dtypes(include=["floating", "integer"]).describe()