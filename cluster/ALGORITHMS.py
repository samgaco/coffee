import os
import pandas as pd
from sklearn import cluster
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
import numpy as np

#set wd
cwd = "/home/samuel/Work/coffe3/data/"
data_sets = os.listdir(cwd)

##Process all objects
data_bus_pro = pd.read_csv(cwd + "/PrePro/preprocessed_data.csv")



def cluster_results(data_frame, name_of_response, k_clusters, SCALE):

    """
    This function applies cluster algorithms over the input data set, and returns metrics, fitted models, and
    the resulting labels.

    :param data_frame: Input data
    :param name_of_response: Name of the response (actual clusters in the data set)
    :param k_clusters: Number of clusters for the algorithm
    :param SCALE: Use scaled data or not
    :return: Returns an object with the fitted algorithms and a list of lists with all the results
    The function it also prints the performance metrics of the clustering algoritgms compared with the
    variable named as name_of_response

    """

    # Cluster
    results = []
    algorithms = {}

    X = StandardScaler().fit_transform(data_frame.drop([name_of_response], axis=1))

    default_base = {'quantile': .3,
                'eps': .3,
                'damping': .9,
                'preference': -200,
                'n_neighbors': 10,
                'n_clusters': 5}

    params = default_base.copy()

    # estimate bandwidth for mean shift
    bandwidth = cluster.estimate_bandwidth(X, quantile=params['quantile'])

    algorithms['kmeans'] = cluster.KMeans(n_clusters=k_clusters, n_init=200)
    algorithms['agglom'] = cluster.AgglomerativeClustering(n_clusters=k_clusters, linkage="ward")
    algorithms['affinity'] = cluster.AffinityPropagation(damping=0.6)
    algorithms["twomeans"] = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
    algorithms["ms"] = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
    algorithms["dbscan"] = cluster.DBSCAN(eps=params['eps'])

    if SCALE:
        for model in algorithms.values():
            model.fit(X)
            results.append(list(model.labels_))
    else:
        for model in algorithms.values():
            model.fit(data_frame, y=name_of_response)
            results.append(list(model.labels_))


    #Metrics

    nmi_results = []
    ars_results = []

    y_true_val = data_frame["score"].values

    # Append the results into lists
    for y_pred in results:
        nmi_results.append(normalized_mutual_info_score(y_true_val, y_pred))
        ars_results.append(adjusted_rand_score(y_true_val, y_pred))

    print("NMI_Results: ", nmi_results)
    print("ARS_Results: ", ars_results)

    i = 0
    concat_clusters = data_frame

    for name, dict in algorithms.items():
        label = pd.Series(np.asarray(results[i])).rename(name)
        i += 1
        concat_clusters = pd.concat([concat_clusters, label], axis=1)


    return algorithms, results, concat_clusters





#Use this if you want to save the data as csv's of each of the clusters for each of the algorithms
#  (if the number of clusters is less than a certain threshold)

if __name__ == '__main__':

    algo, results, concat_clusters = cluster_results(data_bus_pro, "score", 5, True)

    GET_DATA_SETS = False
    FOLDER = "/home/samuel/Work/coffe3/CLUSTER/Results/"

    if GET_DATA_SETS:
        for name, dict in algo.items():
            if len(concat_clusters[name].unique())<10:
                for cluster in concat_clusters[name].unique():
                     concat_clusters[concat_clusters[name]==cluster].to_csv(
                        FOLDER +  name + "_cluster" + str(cluster) + ".csv")

