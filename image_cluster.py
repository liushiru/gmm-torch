######## loading external package dependency ####################
import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal
import imageio
from functools import reduce
import torch
import os

from utils import check_mnist_dataset_exists


def get_image_feature(path):
    Im = imageio.imread(os.path.join(path), pilmode='RGB')
    temp = Im/255. # divide by 255 to get in fraction
    mn = temp.sum(axis=0).sum(axis=0)/(temp.shape[0]*temp.shape[1])
    return mn/np.linalg.norm(mn, ord=None) # taking 2nd norm to scale vector

# data (numpy array) : array of observations
# weights (numpy array) : numpy array of weight of each clusters of size (1, n_clusters)
#means (numpy array) : numpy array of means of each clusters of size (n_cluster, dimension)
#covariances(numpy array) : numpy array of covariance metrix of size (n_clusters, dimension, dimension)
def get_responsibilities( data, weights, means, covariances):
    n_data = len(data)
    n_clusters = len(means)
    resp = np.zeros((n_data, n_clusters))
    for i in range(n_data):
       for k in range(n_clusters):
          resp[i, k] = weights[k]*   multivariate_normal.pdf(data[i],means[k],covariances[k],allow_singular=True)
        # Add up responsibilities over each data point and normalize
    row_sums = resp.sum(axis=1)[:, np.newaxis]
    resp = resp / row_sums
    return resp

# resp(numpy array) : responsibility numpy array size (n_sample, n_clusters)
def get_soft_counts(resp):
    return np.sum(resp, axis=0)
# counts (numpy array) : count list of sum of soft counts for all clusters of size (n_cluster)
def get_weights(counts):
    n_clusters = len(counts)
    sum_count = np.sum(counts)
    weights = np.array(list(map(lambda k : counts[k]/sum_count, range(n_clusters))))
    return weights

def get_kmeans_mu(x, n_centers, init_times=50, min_delta=1e-3):
    """
    Find an initial value for the mean. Requires a threshold min_delta for the k-means algorithm to stop iterating.
    The algorithm is repeated init_times often, after which the best centerpoint is returned.
    args:
        x:            torch.FloatTensor (n, d) or (n, 1, d)
        init_times:   init
        min_delta:    int
    """
    if len(x.size()) == 3:
        x = x.squeeze(1)
    x_min, x_max = x.min(), x.max()
    x = (x - x_min) / (x_max - x_min)

    min_cost = np.inf

    for i in range(init_times):
        tmp_center = x[np.random.choice(np.arange(x.shape[0]), size=n_centers, replace=False), ...]
        l2_dis = torch.norm((x.unsqueeze(1).repeat(1, n_centers, 1) - tmp_center), p=2, dim=2)
        l2_cls = torch.argmin(l2_dis, dim=1)

        cost = 0
        for c in range(n_centers):
            cost += torch.norm(x[l2_cls == c] - tmp_center[c], p=2, dim=1).mean()

        if cost < min_cost:
            min_cost = cost
            center = tmp_center

    delta = np.inf

    while delta > min_delta:
        l2_dis = torch.norm((x.unsqueeze(1).repeat(1, n_centers, 1) - center), p=2, dim=2)
        l2_cls = torch.argmin(l2_dis, dim=1)
        center_old = center.clone()

        for c in range(n_centers):
            center[c] = x[l2_cls == c].mean(dim=0)

        delta = torch.norm((center_old - center), dim=1).max()

    return (center.unsqueeze(0) * (x_max - x_min) + x_min)

if __name__ == "__main__":
    data_path = check_mnist_dataset_exists()
    train_data = torch.load(data_path + 'mnist/train_data.pt')
    train_label = torch.load(data_path + 'mnist/train_label.pt')
    test_data = torch.load(data_path + 'mnist/test_data.pt')
    test_label = torch.load(data_path + 'mnist/test_label.pt')

    train_data = train_data.reshape(train_data.size(0), train_data.size(1)*train_data.size(2))
    val_data = train_data[5000:]
    train_data = train_data[:50000]
    test_data = test_data.reshape(test_data.size(0), test_data.size(1)*test_data.size(2))

    val_label = train_label[50000:]
    train_label = train_label[:50000]

    means = get_kmeans_mu(train_data, 10, init_times=50, min_delta=1e-3)
    vars = torch.nn.Parameter(torch.ones(1, 10, 784), requires_grad=False)