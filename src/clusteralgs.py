"""
Several variants of spectral clustering that we would like to compare.
"""
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics.pairwise import rbf_kernel
import scipy.linalg
import scipy.sparse.linalg

import datasets
import fsg_internal
import math
import numpy as np
import faiss
import stag.graph


def swig_sprs_to_scipy(swig_mat):
    """
    Take a swig sparse matrix and convert it to a scipy sparse matrix
    """
    outer_starts = fsg_internal.sprsMatOuterStarts(swig_mat)
    inner_indices = fsg_internal.sprsMatInnerIndices(swig_mat)
    values = fsg_internal.sprsMatValues(swig_mat)
    return scipy.sparse.csc_matrix((values, inner_indices, outer_starts))


def labels_to_clusters(labels, k=None):
    """Take a list of labels, and return a list of clusters, using the indices"""
    if k is None:
        k = max(labels) + 1

    clusters = [[] for i in range(k)]
    for i, c in enumerate(labels):
        clusters[c].append(i)

    return clusters


def clusters_to_labels(clusters):
    """Take a list of clusters, and return a list of labels"""
    n = sum([len(cluster) for cluster in clusters])
    labels = [0] * n
    for c_idx, cluster in enumerate(clusters):
        for j in cluster:
            labels[j] = c_idx
    return labels


def kmeans(data, k):
    """
    Apply the kmeans algorithm to the given data, and return the labels.
    """
    kmeans_obj = KMeans(n_clusters=k)
    kmeans_obj.fit(data)
    return kmeans_obj.labels_, kmeans_obj.cluster_centers_


def rbf_spectralcluster(dataset: datasets.Dataset, k, gamma=1.0):
    """
    Run the default RBF spectral clustering from sklearn.
    """
    sc = SpectralClustering(n_clusters=k, gamma=gamma)
    sc.fit(dataset.raw_data)

    return sc.labels_


def knn_spectralcluster(dataset: datasets.Dataset, k):
    """
    Run the default knn spectral clustering from sklearn.
    """
    sc = SpectralClustering(n_clusters=k, affinity='nearest_neighbors')
    sc.fit(dataset.raw_data)
    return sc.labels_


def fast_spectral_cluster_ifgt(dataset: datasets.Dataset, k, gamma=1.0):
    """
    Run our new fast spectral clustering algorithm.
    """
    kdesolver = fsg_internal.IFGT()
    adj_mat = fsg_internal.fast_similarity_graph(dataset.raw_data, gamma, kdesolver)
    lap_mat = swig_sprs_to_scipy(fsg_internal.adjacencyToLaplacian(adj_mat))
    _, eigenvectors = scipy.sparse.linalg.eigsh(lap_mat, k, which='SM')
    labels = KMeans(n_clusters=k).fit_predict(eigenvectors)
    return labels


def sample_neighbour(adj, i):
    deg_i = np.sum(adj[i])
    sampled_neighbour = np.random.choice(np.arange(0, adj.shape[0]),
                                         p=adj[i]/deg_i)
    return sampled_neighbour


def sample_neighbours(adj, i, num):
    deg_i = np.sum(adj[i])
    sampled_neighbours = np.random.choice(np.arange(0, adj.shape[0]),
                                          p=adj[i]/deg_i,
                                          size=num)
    return sampled_neighbours


def construct_sz_graph(dataset: datasets.Dataset, gamma=1.0):
    n = dataset.num_data_points
    kernel_matrix = rbf_kernel(dataset.raw_data, gamma=gamma)

    adj_mat = scipy.sparse.lil_matrix((n, n))
    for i in range(n):
        neighbours = sample_neighbours(kernel_matrix, i, int(10 * math.log(n)))
        for neighbour in neighbours:
            if neighbour != i:
                adj_mat[i, neighbour] = 1
                adj_mat[neighbour, i] = 1
    return stag.graph.Graph(adj_mat)


def sz_spectral_cluster(dataset: datasets.Dataset, k, gamma=1.0):
    g = construct_sz_graph(dataset, gamma=gamma)
    lap_mat = g.laplacian()
    _, eigenvectors = scipy.sparse.linalg.eigsh(lap_mat, k, which='SM')
    labels = KMeans(n_clusters=k).fit_predict(eigenvectors)
    return labels


def faiss_exact_spectral_cluster(dataset: datasets.Dataset, k: int):
    n, d = dataset.raw_data.shape
    index = faiss.IndexFlatL2(d)
    index.add(dataset.raw_data)
    k_for_knn = 10
    _, I = index.search(dataset.raw_data, k_for_knn)
    adj_mat = scipy.sparse.lil_matrix((n, n))
    for i in range(n):
        for j in range(1, k_for_knn):
            adj_mat[i, I[i, j]] = -1
            adj_mat[I[i, j], i] = -1
    g = stag.graph.Graph(adj_mat)
    lap_mat = g.laplacian()
    _, eigenvectors = scipy.sparse.linalg.eigsh(lap_mat, k, which='SM')
    labels = KMeans(n_clusters=k).fit_predict(eigenvectors)
    return labels


def faiss_ivf_spectral_cluster(dataset: datasets.Dataset, k: int):
    n, d = dataset.raw_data.shape
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, 1)
    index.train(dataset.raw_data)
    index.add(dataset.raw_data)
    k_for_knn = 10
    _, I = index.search(dataset.raw_data, k_for_knn)
    adj_mat = scipy.sparse.lil_matrix((n, n))
    for i in range(n):
        for j in range(1, k_for_knn):
            adj_mat[i, I[i, j]] = -1
            adj_mat[I[i, j], i] = -1
    g = stag.graph.Graph(adj_mat)
    lap_mat = g.laplacian()
    _, eigenvectors = scipy.sparse.linalg.eigsh(lap_mat, k, which='SM')
    labels = KMeans(n_clusters=k).fit_predict(eigenvectors)
    return labels


def faiss_hnsw_spectral_cluster(dataset: datasets.Dataset, k: int):
    n, d = dataset.raw_data.shape
    index = faiss.IndexHNSWFlat(d, 32)
    index.train(dataset.raw_data)
    index.add(dataset.raw_data)
    k_for_knn = 10
    _, I = index.search(dataset.raw_data, k_for_knn)
    adj_mat = scipy.sparse.lil_matrix((n, n))
    for i in range(n):
        for j in range(1, k_for_knn):
            adj_mat[i, I[i, j]] = -1
            adj_mat[I[i, j], i] = -1
    g = stag.graph.Graph(adj_mat)
    lap_mat = g.laplacian()
    _, eigenvectors = scipy.sparse.linalg.eigsh(lap_mat, k, which='SM')
    labels = KMeans(n_clusters=k).fit_predict(eigenvectors)
    return labels

