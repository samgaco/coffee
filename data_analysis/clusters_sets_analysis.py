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

# we get the concat_clusters, the data frame with the labels of the clusters created by the algorithms concatenated
algo, results, concat_clusters = cluster_results(data_bus_pro, "score", 5, True)

# we create a variable with the names of the algorithms used (they are also names of columns in concat_clusters)

names_algo = []
for name, dict_ in algo.items():
    names_algo.append(name)

plt.ioff()


for i in range(len(names_algo)):

    clusters = concat_clusters[names_algo[i]].unique()

    for j in range(len(clusters)):

        cluster_data = concat_clusters[concat_clusters[names_algo[i]] == clusters[j]]

        cluster_data_corr = cluster_data.corr()

        for correlation in np.linspace(0, 1, 750):
            cd_corr = cluster_data_corr[cluster_data_corr["score"] > correlation]
            if len(cd_corr.index) < 10:
                break

        cluster_data_selection = cluster_data[cd_corr.index]

        path = "/home/samuel/Work/coffe3/01_EDA/clusters_sets_analysis_results/"

        if not os.path.exists(path + names_algo[i]):
            os.makedirs(path + names_algo[i])

        if not os.path.exists(path + names_algo[i] + "/" + str(clusters[j])):
            os.makedirs(path + names_algo[i] + "/" + str(clusters[j]))

        for col in list(cluster_data_selection.drop("score", axis=1)):

            try:
                    g = sns.FacetGrid(cluster_data_selection, col=None, hue="score", palette="husl")
                    g = (g.map(sns.distplot, col, hist=False))
                    plt.legend()

            except Exception:

                print("Singular matrix: " + col)

            g.savefig(
                path + names_algo[i] + "/" + str(clusters[j]) + "/" + col + "_" + names_algo[i] + str(clusters[j]) + ".png")
