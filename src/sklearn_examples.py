import os
import time
import warnings

import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice

import clusteralgs
import datasets as ds

# ============
# Generate datasets. We choose the size big enough to see the scalability
# of the algorithms, but not too big to avoid too long running times
# ============
n_samples = 15000
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05)
blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
no_structure = np.random.rand(n_samples, 2), None

# Anisotropicly distributed data
X, y = datasets.make_blobs(n_samples=n_samples,
                           centers=[[0,0],[10,10],[10,23]])
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)

# ============
# Set up cluster parameters
# ============
plt.figure(figsize=(7, 4))
plt.subplots_adjust(
    left=0.02, right=0.98, bottom=0.01, top=0.95, wspace=0.05, hspace=0.05
)
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Times"
})

plot_num = 1

default_base = {
    "quantile": 0.3,
    "eps": 0.3,
    "damping": 0.9,
    "preference": -200,
    "n_neighbors": 3,
    "n_clusters": 3,
    "min_samples": 7,
    "xi": 0.05,
    "min_cluster_size": 0.1,
}

datasets = [
    (
        noisy_circles,
        {
            "eps": 0.1,
            "damping": 0.77,
            "preference": -240,
            "quantile": 0.2,
            "n_clusters": 2,
            "min_samples": 7,
            "xi": 0.08,
            "gamma": 0.2,
            "rbf_gamma": 10,
        },
    ),
    (
        noisy_moons,
        {
            "damping": 0.75,
            "preference": -220,
            "n_clusters": 2,
            "min_samples": 7,
            "xi": 0.1,
            "gamma": 0.2,
            "rbf_gamma": 5,
        },
    ),
    (
        aniso,
        {
            "eps": 0.15,
            "n_neighbors": 2,
            "min_samples": 7,
            "xi": 0.1,
            "min_cluster_size": 0.2,
            "gamma": 0.2,
            "rbf_gamma": 5,
        },
    ),
]

for i_dataset, (dataset, algo_params) in enumerate(datasets):
    # update parameters with dataset-specific values
    params = default_base.copy()
    params.update(algo_params)

    X, y = dataset

    # normalize dataset for easier parameter selection
    X = StandardScaler().fit_transform(X)

    # estimate bandwidth for mean shift
    bandwidth = cluster.estimate_bandwidth(X, quantile=params["quantile"])

    # connectivity matrix for structured Ward
    connectivity = kneighbors_graph(
        X, n_neighbors=params["n_neighbors"], include_self=False
    )
    # make connectivity symmetric
    connectivity = 0.5 * (connectivity + connectivity.T)

    # ============
    # Create cluster objects
    # ============
    ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
    two_means = cluster.MiniBatchKMeans(n_clusters=params["n_clusters"], n_init="auto")
    ward = cluster.AgglomerativeClustering(
        n_clusters=params["n_clusters"], linkage="ward", connectivity=connectivity
    )
    spectral = cluster.SpectralClustering(
        n_clusters=params["n_clusters"],
        eigen_solver="arpack",
        affinity="nearest_neighbors",
    )
    spectral_rbf = cluster.SpectralClustering(
        n_clusters=params["n_clusters"],
        eigen_solver="arpack",
        affinity="rbf",
        gamma=params["rbf_gamma"],
    )
    dbscan = cluster.DBSCAN(eps=params["eps"])
    optics = cluster.OPTICS(
        min_samples=params["min_samples"],
        xi=params["xi"],
        min_cluster_size=params["min_cluster_size"],
    )
    affinity_propagation = cluster.AffinityPropagation(
        damping=params["damping"], preference=params["preference"], random_state=0
    )
    average_linkage = cluster.AgglomerativeClustering(
        linkage="average",
        metric="cityblock",
        n_clusters=params["n_clusters"],
        connectivity=connectivity,
    )
    birch = cluster.Birch(n_clusters=params["n_clusters"])
    gmm = mixture.GaussianMixture(
        n_components=params["n_clusters"], covariance_type="full"
    )

    clustering_algorithms = (
        # ("MiniBatch\nKMeans", two_means),
        # ("Affinity\nPropagation", affinity_propagation),
        # ("MeanShift", ms),
        ("SKLearn $k$-NN", spectral),
        ("SKLearn GK", spectral_rbf),
        # ("Ward", ward),
        # ("Agglomerative\nClustering", average_linkage),
        # ("DBSCAN", dbscan),
        # ("OPTICS", optics),
        # ("BIRCH", birch),
        # ("Gaussian\nMixture", gmm),
        ("FAISS Exact", None),
        ("FAISS HNSW", None),
        ("FAISS IVF", None),
        ("Our Algorithm", None)
    )

    for name, algorithm in clustering_algorithms:
        print(f"Running {name} on dataset {i_dataset}...")
        t0 = time.time()

        if name == "Our Algorithm":
            dataset = ds.Dataset(raw_data=X)
            y_pred = clusteralgs.fast_spectral_cluster_ifgt(
                dataset, params["n_clusters"], params['gamma'])
        elif name == "FAISS Exact":
            dataset = ds.Dataset(raw_data=X)
            y_pred = clusteralgs.faiss_exact_spectral_cluster(
                dataset, params["n_clusters"])
        elif name == "FAISS HNSW":
            dataset = ds.Dataset(raw_data=X)
            y_pred = clusteralgs.faiss_hnsw_spectral_cluster(
                dataset, params["n_clusters"])
        elif name == "FAISS IVF":
            dataset = ds.Dataset(raw_data=X)
            y_pred = clusteralgs.faiss_ivf_spectral_cluster(
                dataset, params["n_clusters"])
        else:
            # catch warnings related to kneighbors_graph
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="the number of connected components of the "
                    + "connectivity matrix is [0-9]{1,2}"
                    + " > 1. Completing it to avoid stopping the tree early.",
                    category=UserWarning,
                )
                warnings.filterwarnings(
                    "ignore",
                    message="Graph is not fully connected, spectral embedding"
                    + " may not work as expected.",
                    category=UserWarning,
                )
                algorithm.fit(X)

            if hasattr(algorithm, "labels_"):
                y_pred = algorithm.labels_.astype(int)
            else:
                y_pred = algorithm.predict(X)

        t1 = time.time()

        plt.subplot(len(datasets), len(clustering_algorithms), plot_num)
        if i_dataset == 0:
            plt.title(f"\\textsc{{{name}}}", size=10)

        colors = np.array(
            list(
                islice(
                    cycle(
                        [
                            "#377eb8",
                            "#ff7f00",
                            "#4daf4a",
                            "#f781bf",
                            "#a65628",
                            "#984ea3",
                            "#999999",
                            "#e41a1c",
                            "#dede00",
                        ]
                    ),
                    int(max(y_pred) + 1),
                )
            )
        )
        # add black color for outliers (if any)
        colors = np.append(colors, ["#000000"])
        plt.scatter(X[:, 0], X[:, 1], s=1, color=colors[y_pred])

        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 2.5)
        plt.xticks(())
        plt.yticks(())
        plt.text(
            0.99,
            0.01,
            ("%.2fs" % (t1 - t0)).lstrip("0"),
            transform=plt.gca().transAxes,
            size=10,
            horizontalalignment="right",
        )
        plot_num += 1

plt.savefig(os.path.join(os.pardir, "results/figures/comparison.png"))
plt.show()
