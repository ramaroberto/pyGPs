__author__ = 'christiaanleysen'

# This example divides a set of timeseries into two clusters of the most similar timeseries using the general 
# model learn over a set of timeseries.
#
# Find more information in the foloowing paper: 
#
# "Energy consumption profiling using Gaussian Processes",
# Christiaan Leysen*, Mathias Verbeke†, Pierre Dagnely†, Wannes Meert* 
# *Dept. Computer Science, KU Leuven, Belgium
# †Data Innovation Team, Sirris, Belgium
# https://lirias.kuleuven.be/bitstream/123456789/550688/1/conf2.pdf
import pyGPs,math
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from pyGPs.Demo.Clustering import pyGP_extension as GPE
import numpy as np
import matplotlib.pyplot as plt
import math
import logging

logger = logging.getLogger("pyGPs.clustering")

def calculateRMSEPyGP(vectorX, vectorY, weighted=True, plot=False):
    """
    calculate the root mean squared error
    Parameters:
    -----------
    vectorX: timestamps of the timeseries
    vectorY: valueSet of the timeseries
    labelList: labels of the timeseries
    weighted: weight RMSE wrt variance of prediction
    Returns:
    --------
    list of (id,rmse) tuples
    """
    #setX = [preprocessing.scale(element )for element in vectorX]
    setY = preprocessing.scale(vectorY,axis=1)

    model = pyGPs.GPR()      # specify model (GP regression)
    k =  pyGPs.cov.Linear() + pyGPs.cov.RBF() # hyperparams will be set with optimizeHyperparameters method
    model.setPrior(kernel=k)

    hyperparams, model2 = GPE.optimizeHyperparameters([0.0000001, 0.0000001, 0.0000001],
                                                      model, vectorX, setY,
                                                      bounds=[(None, 5), (None, 5), (None, 5)],
                                                      method = 'L-BFGS-B')
    print('hyperparameters used:', hyperparams)
    # mean (y_pred) variance (ys2), latent mean (fmu) variance (fs2), log predictive prob (lp)
    y_pred, ys2, fm, fs2, lp = model2.predict(vectorX[0])

    if plot:
        xs = vectorX[0]
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
    for i in range(len(vectorY)):
        if weighted:
            rmse = math.sqrt(mean_squared_error(vectorY[i], y_pred, 1.1*np.max(ys2)-ys2))
        else:
            rmse = math.sqrt(mean_squared_error(vectorY[i], y_pred))
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

    listRMSE, hyperparams, model = calculateRMSEPyGP(vectorX, vectorY, weighted=weighted, plot=plot)
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
        cluster_left = []
        cluster_right = []
        for i, cur_rmse in sortedListRMSE:
            if cur_rmse <= split_rmse:
                cluster_left.append((series[0][i], series[1][i]))
            else:
                cluster_right.append((series[0][i], series[1][i]))
    else:
        print("ERROR: either rmse or clusterSize should be set")
        return None

    if min_size is None or (len(cluster_left) >= min_size and len(cluster_right) >= min_size):
        # check goodness of cluster
        return cluster_left, cluster_right, model, hyperparams
    else:
        logger.debug('Cluster size too small, stopping')
        return series, [], model, hyperparams


def hierarchical_rec(series, max_depth=None, depth=0, **kwargs):

    logger.info("Hierarchical clustering, level {}".format(depth))
    if max_depth is not None and depth >= max_depth:
        return series, None
    cluster1, cluster2, model, hyperparams = hierarchical_step(series, **kwargs)
    if cluster2 == [] or cluster2 is None:
        return cluster1, None, model, hyperparams
    return hierarchical_rec(cluster1, depth + 1, **kwargs), hierarchical_rec(cluster2, depth + 1, **kwargs),\
        model, hyperparams


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

