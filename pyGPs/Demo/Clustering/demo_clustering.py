"""

__author__ = ['christiaanleysen', 'wannesmeert']

This example divides a set of time-series into two clusters of the most similar time-series using the general
model learn over a set of time-series.

Find more information in the following paper:

"Energy consumption profiling using Gaussian Processes",
Christiaan Leysen*, Mathias Verbeke†, Pierre Dagnely†, Wannes Meert*
*Dept. Computer Science, KU Leuven, Belgium
†Data Innovation Team, Sirris, Belgium
https://lirias.kuleuven.be/bitstream/123456789/550688/1/conf2.pdf
"""
import pyGPs
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from pyGPs.Demo.Clustering import pyGP_extension as GPE
import numpy as np
import matplotlib.pyplot as plt
import math
import logging
from collections import namedtuple


logger = logging.getLogger("pyGPs.clustering")


def calculate_rmse_gp(vector_x, vector_y, weighted=True, plot=False):
    """Calculate the root mean squared error.

    :param vector_x: timestamps of the timeseries
    :param vector_y: valueSet of the timeseries
    :param weighted: weight RMSE wrt variance of prediction
    :param plot: plot the expected function
    :returns: list(idx,rmse), hyperparams, model
    """
    # setX = [preprocessing.scale(element )for element in vectorX]
    setY = preprocessing.scale(vector_y, axis=1)

    model = pyGPs.GPR()      # specify model (GP regression)
    k =  pyGPs.cov.Linear() + pyGPs.cov.RBF() # hyperparams will be set with optimizeHyperparameters method
    model.setPrior(kernel=k)

    hyperparams, model2 = GPE.optimizeHyperparameters([0.0000001, 0.0000001, 0.0000001],
                                                      model, vector_x, setY,
                                                      bounds=[(None, 5), (None, 5), (None, 5)],
                                                      method = 'L-BFGS-B')
    print('hyperparameters used:', hyperparams)
    # mean (y_pred) variance (ys2), latent mean (fmu) variance (fs2), log predictive prob (lp)
    y_pred, ys2, fm, fs2, lp = model2.predict(vector_x[0])

    if plot:
        xs = vector_x[0]
        ym = y_pred
        xss = np.reshape(xs, (xs.shape[0],))
        ymm = np.reshape(ym, (ym.shape[0],))
        ys22 = np.reshape(ys2, (ys2.shape[0],))
        for i in setY:
            plt.plot(i,color='blue')
        plt.plot(xss, ym, color='red', label="Prediction")
        plt.fill_between(xss, ymm + 2. * np.sqrt(ys22), ymm - 2. * np.sqrt(ys22),
                         facecolor=[0.7539, 0.89453125, 0.62890625, 1.0], linewidths=0.0, alpha=0.5)
        plt.legend()
        plt.show(block=True)

    rmseData = []
    for i in range(len(vector_y)):
        if weighted:
            rmse = math.sqrt(mean_squared_error(vector_y[i], y_pred, 1.1 * np.max(ys2) - ys2))
        else:
            rmse = math.sqrt(mean_squared_error(vector_y[i], y_pred))
        rmseData.append((i,rmse))
    return rmseData, hyperparams, model2


def hierarchical_step(series, split_rmse=None, max_avgrmse=None, min_size=None, split_ratio=None,
                      weighted=True, plot=False):
    """
    aux method for the clustering which divides the clusterlist further into clusters using a certain threshold.

    :param series: list with timeseries which needs to be clustered
    :param split_rmse: Split on this rmse (optional)
    :param max_avgrmse: mean similarity threshold to divide the clusters, otherwise do not split
    :param min_size: minimum cluster size, otherwise do not split
    :param splitratio: ratio of timeseries that will be devided into the left and right cluster (optional)
    :returns: (series_left, series_right, model, hyperparams)
    """
    vectorX, vectorY = series

    listRMSE, hyperparams, model = calculate_rmse_gp(vectorX, vectorY, weighted=weighted, plot=plot)
    sortedListRMSE = sorted(listRMSE, key=lambda x: x[1])
    mean_rmse = np.mean([t[1] for t in sortedListRMSE])
    logger.info("Split at node, RMSE = [{}, {}, {}]".format(sortedListRMSE[0][1], mean_rmse, sortedListRMSE[-1][1]))

    if max_avgrmse is not None and mean_rmse < max_avgrmse:
        return series, [], model, hyperparams

    # NormalizeValue = sortedListRMSE[-1][1]
    # sortedListRMSE_normalized = [(x[0], x[1] / NormalizeValue) for x in sortedListRMSE][::-1]

    if min_size is not None:
        clusterSizeLength = int(math.ceil(split_ratio * len(sortedListRMSE)))
        cluster_left = sortedListRMSE[-clusterSizeLength:]#[::-1]
        cluster_right = sortedListRMSE[:len(sortedListRMSE)-clusterSizeLength]#[::-1]
    elif split_rmse is not None:
        cluster_left_x = []
        cluster_left_y = []
        cluster_right_x = []
        cluster_right_y = []
        for i, cur_rmse in sortedListRMSE:
            if cur_rmse <= split_rmse:
                cluster_left_x.append(series[0][i])
                cluster_left_y.append(series[1][i])
            else:
                cluster_right_x.append(series[0][i])
                cluster_right_y.append(series[1][i])
        cluster_left = (cluster_left_x, cluster_left_y)
        cluster_right = (cluster_right_x, cluster_right_y)
    else:
        print("ERROR: either rmse or clusterSize should be set")
        return None

    if min_size is None or (len(cluster_left[0]) >= min_size and len(cluster_right[0]) >= min_size):
        # check goodness of cluster
        return cluster_left, cluster_right, model, hyperparams
    else:
        logger.debug('Cluster size too small, stopping')
        return series, [], model, hyperparams


ClusterNode = namedtuple("ClusterNode", ["left", "right", "model", "hyperparameters", "depth"])
ClusterLeaf = namedtuple("ClusterLeaf", ["series", "depth"])


def hierarchical_rec(series, max_depth=None, depth=0, **kwargs):
    logger.info("Hierarchical clustering, level {}".format(depth))
    if max_depth is not None and depth >= max_depth:
        return ClusterLeaf(series, depth)
    cluster_left, cluster_right, model, hyperparams = hierarchical_step(series, **kwargs)
    if cluster_right == [] or cluster_right is None:
        return ClusterLeaf(cluster_left, depth)
    if cluster_left == [] or cluster_left is None:
        return ClusterLeaf(cluster_right, depth)
    return ClusterNode(hierarchical_rec(cluster_left, depth + 1, **kwargs),
                       hierarchical_rec(cluster_right, depth + 1, **kwargs),
                       model, hyperparams, depth)


def hierarchical(series, max_depth=None, **kwargs):
    """Hierarchical clustering

    :param series: [vectorX, vectorY]
    :param max_depth: Max tree depth
    :param kwargs: Args for divideInClusters
    :return: (series_left, series_right, model, hyperparams)
    """
    return hierarchical_rec(series, max_depth=max_depth, depth=0, **kwargs)


def test():
    vectorX =[]
    vectorY =[]
    # Fill the x-values of the timeseries with a time between 0 and 20
    for i in range(0,4):
        vectorX.append(np.array(range(0,20)))
    print(vectorX)

    # Fill the y-values of the timeseries (actual values)
    vectorY.append(np.array(range(3,23)))
    vectorY.append(np.array(range(4,24)))
    vectorY.append(np.array([2,2,2,2,2,2,2,2,2,2,2,13,14,15,16,17,18,19,20,21]))
    vectorY.append(np.array([3,3,3,3,3,3,3,3,3,3,3,11,12,13,14,15,16,17,18,19]))

    # show input timeseries
    label = 0
    for i in vectorY:
        plt.plot(i,label='timeries'+str(label))
        label += 1
    plt.legend()
    plt.show()

    # choose cluster parameters
    splitRatio = 0.5 # splitsings ratio of the clusters
    minClusterSize=1 # Minimum cluster size
    meanSimilarityThreshold = 0.9 # similarity threshold

    newClusterlist,newRemaininglist = hierarchical_step([vectorX, vectorY], None, meanSimilarityThreshold,
                                                        minClusterSize, splitRatio)

    print("cluster1: "+str(np.sort(newClusterlist)))
    print("cluster2: "+str(np.sort(newRemaininglist)))


if __name__ == "__main__":
    test()

