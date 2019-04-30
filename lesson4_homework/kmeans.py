'''
author:huangchao
considering:
1.implement by vectorization
2.converge condition
  if change of each center axis is less than 0.001 or it reach iterate_num
3.configure init_num param for run_kms(),
  it will again choose initial blobs to cluster for at most init_num times ,if current initial blobs can't converge
4. run_kms() considered compatible with multiple num of cluster center and blobs features
'''

import numpy as np
import numba as nb
import sklearn.datasets as db
import matplotlib.pyplot as plt


def one_to_all_distance(A_array, target):
    '''
    A_array : array of shape [n_samples, n_features]
    target: array of shape [1, n_features]
    '''
    # only support array calculation
    try:
        assert (type(A_array) == np.ndarray and type(target) == np.ndarray)
    except Exception:
        raise TypeError("both input type must be array")

    # sample attribute must be the same
    try:
        assert (A_array.shape[1] == target.shape[1])
    except Exception:
        raise AttributeError("n_features must be the same")

    dis = np.linalg.norm(A_array - target, axis=1, keepdims=True)
    assert (dis.shape == (A_array.shape[0], 1))
    return dis


# @nb.jit(nopython=True)
def run_kms(X, y, init_num=10, iterate_num=20):
    '''
    X : array of shape [n_samples, n_features]
        The generated samples.
    y: array of shape [n_samples,1]
        The integer labels for cluster membership of each sample.
    init_num:int,optional(default=10)
        Indicator of repeated times of cluster with different initial blobs
    iter_num:int,optional(default=10)
        max iterate num
    '''
    n_centers = len(np.unique(y))
    n_samples = X.shape[0]
    n_features = X.shape[1]
    print("n_center:{},n_sample:{},n_features:{}".format(n_centers, n_samples, n_features))
    cluster_blobs = np.zeros((n_centers, n_features))
    prev_blobs = cluster_blobs

    # choose initial center point
    for i in range(init_num):
        center_index = np.random.choice(n_samples, size=n_centers, replace=False)
        blobs = X[center_index]
        aver_dis_center_to_samples = np.zeros((1, n_centers))
        # stop judgement
        for k in range(iterate_num):
            # calculate distance of center to all samples
            dis_center_to_samples = np.zeros((n_samples, 1))
            for j in range(n_centers):
                dis = one_to_all_distance(X, blobs[j].reshape(1, n_features))
                dis_center_to_samples = np.hstack((dis_center_to_samples, dis))
            dis_center_to_samples = np.delete(dis_center_to_samples, 0, axis=1)
            assert (dis_center_to_samples.shape == (n_samples, n_centers))
            # choose the center samples belong to and recaculate the new center
            center_index_of_samples = dis_center_to_samples.argmin(axis=1)
            assert (center_index_of_samples.shape == (n_samples,))
            for j in range(n_centers):
                blobs[j] = np.mean(X[center_index_of_samples == j], axis=0)
                aver_dis_center_to_samples[0][j] = np.mean(dis_center_to_samples[center_index_of_samples == j][j],
                                                           axis=0)
                print("initial:{} ,iterate:{},average distance {} from center {} to samples ".format(i, k, aver_dis_center_to_samples[0][j], j))
            assert (blobs.shape == (n_centers, n_features))

            # judge whether stop cluster
            converge_flag = True
            for f in range(n_centers):
                for g in range(n_features):
                    converge_flag = converge_flag and np.abs(blobs[f][g] - prev_blobs[f][g]) < 0.001
            prev_blobs = np.copy(blobs)
            if converge_flag == True:
                cluster_blobs = np.copy(blobs)
                print("initial:{} ,iterate:{},it's converged".format(i, k))
                return cluster_blobs

    return blobs


if __name__ == "__main__":
    X, y = db.make_blobs(n_samples=[100, 200, 300], centers=None, cluster_std=5.0, center_box=(-20, 20), shuffle=True,
                         random_state=1)
    print("X shape={},y shape={}".format(X.shape, y.shape))

    cluster = run_kms(X, y, init_num=5, iterate_num=10)

    #plot
    plt.figure(1)
    plt.scatter(X[y == 0, 0], X[y == 0, 1], marker="+", c='k')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], marker="o", c='b')
    plt.scatter(X[y == 2, 0], X[y == 2, 1], marker="+", c='g')
    for i in range(cluster.shape[0]):
        plt.scatter(cluster[i, 0], cluster[i, 1], marker="*", c='r')
    plt.title("cluster of blobs")
    plt.show()
