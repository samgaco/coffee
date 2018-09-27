import os
import pandas as pd
from CLUSTER.ALGORITHMS import cluster_results
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


# set wd
cwd = "/home/samuel/Work/coffe3/data/"
data_sets = os.listdir(cwd)

##Process all objects
data_bus_pro = pd.read_csv(cwd + "/PrePro/preprocessed_data.csv")


#Let's work on different feature selections, this time we will eliminate all variables containing the word google in the name
#and we will get only the variables correlated with the scores above a certain threshold.


google_popular_names = list(data_bus_pro.filter(regex = "google"))
data_bus_pro_without_google = data_bus_pro.drop(google_popular_names, axis= 1)

thres = 0.06
db_corr = data_bus_pro_without_google.corr()
db_corr_sel = db_corr[db_corr["score"] > thres]

data_bus_pro_thres = data_bus_pro[list(db_corr_sel.index)]

data_bus_pro_thres.to_csv("/home/samuel/Work/coffe3/CLUSTER/selection_of_features_results/selection_feat" + str(thres) + ".csv")

# we get the concat_clusters, the data frame with the labels of the clusters created by the algorithms concatenated
algo, results, concat_clusters = cluster_results(data_bus_pro_thres, "score", 5, True)

