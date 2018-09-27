import os
import pandas as pd
from CLUSTER.ALGORITHMS import cluster_results
from data_analysis.pca_clustering import plotly_clusters
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#set wd
cwd = "/home/samuel/Work/coffe3/data/"
data_sets = os.listdir(cwd)

##Process all objects
data_bus_pro = pd.read_csv(cwd + "/PrePro/preprocessed_data.csv")


#Let's see what clusters we get by selecting just two and let's try to understand the differences.
algo, results, concat_clusters = cluster_results(data_bus_pro, "score", 2, True)

#plotly_clusters(data_bus_pro, "score", results[1])
#plotly_clusters(data_bus_pro, "score", results[0])

#We select K-means because it seems more structured and defined

first_cluster_1 = concat_clusters[concat_clusters["kmeans"] == 1].drop(["agglom", "affinity", "twomeans", "ms", "dbscan"], axis=1)
first_cluster_0 = concat_clusters[concat_clusters["kmeans"] == 0].drop(["agglom", "affinity", "twomeans", "ms", "dbscan"], axis=1)




def binary_nodiffer(first_cluster_0, first_cluster_1, concat_clusters, method, BINARY_DIFFER):

    """

    Searchs for the binary columns that do not differ completely (all 1's or all 0's), and prompts back
    a table with the means to compare the columns in each cluster and how to they differ.

    :param first_cluster_0: Input one cluster
    :param first_cluster_1: Input second cluster
    :param concat_clusters: input output of cluster_results "concat_clusters"
    :param method: clustering method you are analyzing
    :return:

    """

    if not BINARY_DIFFER:

        array01 = np.array([1,0])
        all_01_differ = []
        for feature in list(first_cluster_1):
            if feature in list(first_cluster_0):
                if np.array_equal(first_cluster_0[feature].unique(), array01) and np.array_equal(first_cluster_1[feature].unique() , array01):
                    all_01_differ.append(feature)
        print("Number of binary variables which do not differ completely: ", len(all_01_differ))

        compare_clusters_selected = concat_clusters.groupby(method)[all_01_differ].mean()

        return all_01_differ, compare_clusters_selected


    else:
        array1 = np.array[1]
        array0 = np.array[0]

        all_01_differ = []
        for feature in list(first_cluster_1):
            if feature in list(first_cluster_0) and (
                    np.array_equal(first_cluster_0[feature].unique(), array1) and np.array_equal(
                    first_cluster_1[feature].unique(), array0) or np.array_equal(first_cluster_0[feature].unique(),
                                                                                 array0) and np.array_equal(
                    first_cluster_1[feature].unique(), array1)):

                all_01_differ.append(feature)

        compare_clusters_selected = concat_clusters.groupby(method)[all_01_differ].unique()

        return all_01_differ, compare_clusters_selected



if __name__ == '__main__':

    VISUALIZATION = False

    if VISUALIZATION:

        plt.ioff()

        for feature in list(first_cluster_1):
            if feature in list(first_cluster_0):

                try:
                    sns.kdeplot(first_cluster_0[feature], label="first_cluster_0")
                    sns.kdeplot(first_cluster_1[feature], label="first_cluster_1")
                    plt.legend(loc='upper right')
                    plt.show()
                    plt.savefig("/home/samuel/Work/coffe3/CLUSTER/applying_clusters_results/visualizations/first_cluster/" + feature + "_firstcluster.png")

                except Exception:
                    print(feature + " singular matrix :(")
                    np.linspace(100, -100, 100)
                    plt.hist(first_cluster_0[feature], label="first_cluster_0")
                    plt.hist(first_cluster_1[feature], label="first_cluster_1")
                    plt.legend(loc='upper right')
                    plt.show()
                    plt.savefig(
                        "/home/samuel/Work/coffe3/CLUSTER/applying_clusters_results/visualizations/first_cluster/" + feature + "_firstcluster.png")


