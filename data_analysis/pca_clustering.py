from CLUSTER.ALGORITHMS import cluster_results
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import plotly.offline as py
import plotly.graph_objs as go
from sklearn.metrics import confusion_matrix
import seaborn as sns

#set wd
cwd = "/home/samuel/Work/coffe3/data/"
data_sets = os.listdir(cwd)

##Process all objects
data_bus_pro = pd.read_csv(cwd + "/PrePro/preprocessed_data.csv")


def plot_clusters(data_frame, name_score, labels, ThreeDim):

    """
    So this function gets and input data_frame, asks for the name of the response variable in the data_frame,
    so it can get rid off it before applying some PCA on it, and asks for some labels (maybe actual labels
    that the data might come with, like the ones in the column name_score, or just labels gotten from clustering algorithms
    the thing is, labels will be used to color the points in the visualization...also you can decide whether you want ThreeDim (3D plot)
    or not.

    :param data_frame: Data set
    :param name_score: Name of the scores in the data set
    :param labels: Criteria to color the dots
    :param ThreeDim: Do you want 3dimensions OR NOT?!
    :return: Returns a visualization

    """

    labels = pd.DataFrame(labels)
    targets = labels.iloc[:,0].unique()
    colors = ['r', 'g', 'b', "purple", 'orange']

    #Let's plot the data in 2D and color the data points by the scores given by the expert!!!



    if not ThreeDim:

        X = StandardScaler().fit_transform(data_frame.drop([name_score], axis=1))

        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(X)

        principalDf = pd.DataFrame(data = principalComponents
                 , columns = ['principal component 1', 'principal component 2'])


        labels = pd.Series(np.asarray(labels.iloc[:,0])).rename(name_score)
        finalDf = pd.concat([principalDf, labels], axis = 1)


        fig = plt.figure(figsize = (8,8))
        ax = fig.add_subplot(1,1,1)
        ax.set_xlabel('Principal Component 1', fontsize = 15)
        ax.set_ylabel('Principal Component 2', fontsize = 15)
        ax.set_title('2 component PCA', fontsize = 20)

        for target, color in zip(targets, colors):

           indicesToKeep = finalDf[name_score] == target
           ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                   , finalDf.loc[indicesToKeep, 'principal component 2']
                   , c = color
                   , s = 50)
           ax.legend(targets)
           ax.grid()


    #Let's plot the data in 3D and color the data points by the scores given by the expert!!!

    else:

        X = StandardScaler().fit_transform(data_frame.drop([name_score], axis=1))

        pca = PCA(n_components=3)
        principalComponents = pca.fit_transform(X)

        principalDf_3 = pd.DataFrame(data = principalComponents
                 , columns = ['principal component 1', 'principal component 2', 'principal component 3'])



        labels = pd.Series(np.asarray(labels.iloc[:,0])).rename(name_score)

        finalDf_3 = pd.concat([principalDf_3, labels], axis = 1)


        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.set_xlabel('Principal Component 1', fontsize = 15)
        ax.set_ylabel('Principal Component 2', fontsize = 15)
        ax.set_zlabel('Principal Component 3', fontsize = 15)



        ax.scatter( xs = finalDf_3['principal component 1'],
                    ys = finalDf_3['principal component 2'],
                    zs = finalDf_3['principal component 3'],
                    s=9, c=None, depthshade=True)


        for target, color in zip(targets, colors):
            indicesToKeep = finalDf_3[name_score] == target
            ax.scatter(finalDf_3.loc[indicesToKeep, 'principal component 1']
                   , finalDf_3.loc[indicesToKeep, 'principal component 2']
                   , finalDf_3.loc[indicesToKeep, 'principal component 3']
                   , c = color
                   , s = 9)

            ax.legend(targets)
            ax.grid()



##USING THE FUNCTION ABOVE (EXAMPLE):

algo, results = cluster_results(data_bus_pro, "score", 5, True)
plot_clusters(data_bus_pro, "score", results[1], False)



##Okay 3D plots by matplotlib seem to be quite awful and do some funny stuff when trying to change perspective,
#  colors vary like some sort of fucked up confetti spectacle,
##Sooo let's use plotly to see if we can get better conclusions and get a more coherent visualization :)

## PLOTLY

# (code for the visualization completly ripped off from: https://plot.ly/python/3d-scatter-plots/ )


def plotly_clusters(data_frame, name_score, labels):


    X = StandardScaler().fit_transform(data_frame.drop([name_score], axis=1))

    pca = PCA(n_components=3)
    principalComponents = pca.fit_transform(X)

    principalDf_3 = pd.DataFrame(data=principalComponents
                                 , columns=['principal component 1', 'principal component 2', 'principal component 3'])

    labels = pd.DataFrame(labels)
    labels = pd.Series(np.asarray(labels.iloc[:, 0])).rename(name_score)

    finalDf_3 = pd.concat([principalDf_3, labels], axis=1)

    x = finalDf_3.iloc[:,0]
    y = finalDf_3.iloc[:,1]
    z = finalDf_3.iloc[:,2]


    trace1 = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=4,
            color=labels,                # set color to an array/list of desired values
            colorscale='Rainbow',   # choose a colorscale
            opacity=0.8
        )
    )

    data = [trace1]
    layout = go.Layout(
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        )
    )
    fig = go.Figure(data=data, layout=layout)
    py.plot(fig, filename='3d-scatter-colorscale')


plotly_clusters(data_bus_pro, "score", results[1])




##Let's get a confusion matrix
#we can decide which predicted clusters adjust the best to the actual "clusters" (scores)

def confusion_clusters(data_bus_pro_scores, labels):

    datascore = np.asarray(data_bus_pro_scores)
    labels = np.asarray(labels) + 1

    mat = confusion_matrix( datascore , labels )
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
    plt.xlabel('true label')
    plt.ylabel('predicted label')

confusion_clusters(data_bus_pro["score"], results[1])

