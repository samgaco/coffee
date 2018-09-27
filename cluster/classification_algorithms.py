import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
import random
from sklearn.metrics import mean_squared_error

#set wd
cwd = "/home/samuel/Work/coffe3/data/"
data_sets = os.listdir(cwd)

##Process all objects
data_bus_pro = pd.read_csv(cwd + "/PrePro/preprocessed_data.csv")


#we make use of this two algorithms with the expectation of making use of the results to figure out more things about how
#our data is clutered, or should be

def pattern_algo(data_frame, name_of_response , n_neighbors):

    """

    :param data_frame:
    :param name_of_response:
    :param n_neighbors:
    :return:

    """

    X = StandardScaler().fit_transform(data_frame.drop([name_of_response], axis=1))

    random.seed(123)
    x_train, x_test, y_train, y_test = train_test_split(X, data_frame[name_of_response], random_state=0)

    algorithms = {'knn': KNeighborsClassifier(n_neighbors=n_neighbors),
                  'lof': LocalOutlierFactor(n_neighbors=n_neighbors)}


    results = pd.DataFrame()
    for algorithm, dict in algorithms.items():

        algorithms[algorithm].fit(x_train, y_train)

        if algorithm == "lof":
            results[algorithm] = algorithms[algorithm]._predict(x_test)
        else:
            results[algorithm] = algorithms[algorithm].predict(x_test)

        print(algorithm + ":t", mean_squared_error(results[algorithm], y_test))


    return results

