import os
import pandas as pd
import matplotlib as plt
import seaborn as sns
from sklearn.cluster import KMeans


#set wd
cwd = "/home/samuel/Work/coffe3/data/"
data_sets = os.listdir(cwd)

#select and read our data
mask = ["businesses" in x for x in data_sets]
data_bus_read = data_sets[3]

data_bus = pd.read_csv(cwd + data_bus_read, engine="python")

#data exploration
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)
data_bus
data_bus.shape

data_bus.dtypes


list(data_bus.select_dtypes(include=["floating", "integer"]))
data_bus.select_dtypes(include=["floating", "integer"]).corr()


#let's analyse the reviews by scores
#first lets check score distribution
data_bus["score"].hist()

#lets see the density of the scores based on photos_counts



data_bus["score"].unique()


def plotdenvsscores(densities, data_bus, title):

    """
    :param densities: Select column from which to plot the densities
    :param data_bus: Data set we are plotting from
    :param title: Title for the graphics
    :return: Returns a plot of the selected density split by the scores

    """

    scores1 = data_bus["score"] == 1
    scores2 = data_bus["score"] == 2
    scores3 = data_bus["score"] == 3
    scores4 = data_bus["score"] == 4
    scores5 = data_bus["score"] == 5

    data_s1 = data_bus[scores1]
    data_s2 = data_bus[scores2]
    data_s3 = data_bus[scores3]
    data_s4 = data_bus[scores4]
    data_s5 = data_bus[scores5]


    ax = sns.kdeplot(data_s1[densities], shade=True, color="r", label= "ONE")
    ax = sns.kdeplot(data_s2[densities], shade=True, color="b", label="TWO")
    ax = sns.kdeplot(data_s3[densities], shade=True, color="g", label="THREE")
    ax = sns.kdeplot(data_s4[densities], shade=True, color="purple", label="FOUR")
    ax = sns.kdeplot(data_s5[densities], shade=True, color="orange", label="FIVE")
    ax.legend()
    ax.set_title(title)


plotdenvsscores("photos_count", data_bus ,"Densities of the photos count separated by categories" )
#the counts in the fifth categories tend to be higher than in the other score categories


## Process all objects
data_bus_pro = pd.read_csv(cwd + "/PrePro/preprocessed_data.csv")

data_bus_pro_corr = data_bus_pro.corr()
data_bus_pro_corr.to_csv("/home/samuel/Work/coffe3/01_EDA/02_EDA_Results/Descriptives/preprocessed_corr.csv")

def corrs_frame(data_frame, correlation_threshold):

    """
    :param data_frame: Input data frame you want to get the correlations from
    :param correlation_threshold: Input the threshold from which to select the features to put in the correlation
    matrix
    :return: Returns a data frame of correlations based on the input threshold
    """
    data_bus_pro_corr = data_frame.corr()
    data_corr_threshold = data_bus_pro_corr[data_bus_pro_corr["score"]>=correlation_threshold]
    data_corr_threshold = data_corr_threshold[data_corr_threshold.index]

    return data_corr_threshold

corr0 = corrs_frame(data_bus_pro, 0.23)

sns.heatmap(corr0, square = True)

plotdenvsscores("good_for_Brunch", data_bus_pro ,"Densities of Good for Brunch separated by scores" )
plotdenvsscores("good_for_Breakfast", data_bus_pro ,"Densities of good_for_Breakfast separated by scores" )


