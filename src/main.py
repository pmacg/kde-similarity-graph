"""
Run some spectral clustering experiments.
"""
from time import time
import matplotlib.ticker
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
import clusteralgs
import datasets
from sklearn.metrics.pairwise import rbf_kernel
import stag.graph
import fsg_internal
import scipy as sp
import scipy.sparse
import networkx as nx

import warnings
warnings.filterwarnings("ignore")


class ExperimentRunData(object):

    def __init__(self, dataset, running_time, labels, extra_info=None):
        self.running_time = running_time
        self.ari = dataset.ari(labels)
        self.extra_info = extra_info
        self.num_data_points = dataset.num_data_points
        self.dataset = dataset
        self.labels = labels


def two_moons_experiment():
    print("Clustering two moons dataset.")
    k = 2

    algorithms_to_compare = {
        "Scipy RBF": (lambda ds: clusteralgs.rbf_spectralcluster(ds, k, gamma=200)),
        "Scipy KNN": (lambda ds: clusteralgs.knn_spectralcluster(ds, k)),
        "FAISS Exact": (lambda ds: clusteralgs.faiss_exact_spectral_cluster(ds, k)),
        "FAISS HNSW": (lambda ds: clusteralgs.faiss_hnsw_spectral_cluster(ds, k)),
        "FAISS IVF": (lambda ds: clusteralgs.faiss_ivf_spectral_cluster(ds, k)),
        "IFGT FSC": (lambda ds: clusteralgs.fast_spectral_cluster_ifgt(ds, k, 0.1)),
    }

    max_data_size = {
        "Scipy RBF": float('inf'),
        "Scipy KNN": float('inf'),
        "FAISS Exact": float('inf'),
        "FAISS HNSW": float('inf'),
        "FAISS IVF": float('inf'),
        "IFGT FSC": float('inf'),
    }

    max_time_cutoff = {
        "Scipy RBF": 120,
        "Scipy KNN": 120,
        "FAISS Exact": 120,
        "FAISS HNSW": 120,
        "FAISS IVF": 120,
        "IFGT FSC": 120,
    }

    cut_off = {k: False for k in max_time_cutoff}

    experimental_data = {k: [] for k in algorithms_to_compare}
    for na in np.logspace(3, 6, num=30):
        n = int(na)

        print(f"Number of data points = {n}")
        dataset = datasets.TwoMoonsDataset(n=n)

        for alg_name, func in algorithms_to_compare.items():
            if not cut_off[alg_name] and n <= max_data_size[alg_name]:
                tstart = time()
                sc_labels = func(dataset)
                tend = time()
                duration = tend - tstart
                experimental_data[alg_name].append(ExperimentRunData(dataset, duration, sc_labels))
                print(f"{alg_name}: {duration:0.3f} seconds, {experimental_data[alg_name][-1].ari:0.3f} ARI")

                if duration > max_time_cutoff[alg_name]:
                    cut_off[alg_name] = True
            else:
                print(f"Skipping {alg_name}...")

        print("")

    # Save the results
    with open("results/twomoons/results.pickle", 'wb') as fout:
        pickle.dump(experimental_data, fout)


def two_moons_experiment_plot(save=False):
    with open("results/twomoons/results.pickle", 'rb') as fin:
        experimental_data = pickle.load(fin)

    algname_map = {'Scipy RBF': '\\textsc{SKLearn GK}',
                   'Scipy KNN': '\\textsc{SKLearn $k$-NN}',
                   'FAISS Exact': '\\textsc{FAISS Exact}',
                   'FAISS HNSW': '\\textsc{FAISS HNSW}',
                   'FAISS IVF': '\\textsc{FAISS IVF}',
                   'IFGT FSC': '\\textsc{Our Algorithm}',
                   }

    linestyle_map = {'Scipy KNN': 'dashed',
                     'Scipy RBF': 'dotted',
                     'FAISS Exact': 'dashed',
                     'FAISS HNSW': 'dotted',
                     'FAISS IVF': 'solid',
                     'IFGT FSC': 'solid',
                     }

    color_map = {'Scipy KNN': 'blue',
                 'Scipy RBF': 'blue',
                 'FAISS Exact': 'green',
                 'FAISS HNSW': 'green',
                 'FAISS IVF': 'green',
                 'IFGT FSC': 'red',
                 }

    # Display the results
    fig = plt.figure(figsize=(3.25, 2.75))
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Times"
    })
    ax = plt.axes([0.2, 0.16, 0.75, 0.82])
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    for alg_name in linestyle_map:
        data = experimental_data[alg_name]

        plt.plot([d.num_data_points for d in data],
                 [d.running_time for d in data],
                 label=algname_map[alg_name],
                 linewidth=3,
                 linestyle=linestyle_map[alg_name],
                 color=color_map[alg_name])

    plt.legend(loc='best', fontsize=10)
    plt.xlabel('Number of data points', fontsize=10)
    plt.ylabel('Running time (s)', fontsize=10)
    plt.xticks([0, 50000, 100000])
    ax.set_ylim(0, 80)
    ax.set_xlim(0, 100000)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)

    if save:
        plt.savefig("results/figures/twomoons_rebuttal.pdf")

    plt.show()

    # Display the results
    fig = plt.figure(figsize=(3.25, 2.75))
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Times"
    })
    ax = plt.axes([0.2, 0.16, 0.72, 0.82])
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    for alg_name in linestyle_map:
        data = experimental_data[alg_name]

        plt.plot([d.num_data_points for d in data],
                 [d.running_time for d in data],
                 label=algname_map[alg_name],
                 linewidth=3,
                 linestyle=linestyle_map[alg_name],
                 color=color_map[alg_name])

    # plt.legend(loc='best', fontsize=10)
    plt.xlabel('Number of data points', fontsize=10)
    plt.ylabel('Running time (s)', fontsize=10)
    # plt.xticks([0, 50000, 100000])
    ax.set_ylim(0, 80)
    ax.set_xlim(0, 20000)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)

    if save:
        plt.savefig("results/figures/twomoonssmall_rebuttal.pdf")

    plt.show()


def bsds_experiment(image_idx):
    print(f"Clustering BSDS dataset image {image_idx}.")
    gamma = 0.2

    algorithms_to_compare = {
        "Scipy RBF": (lambda ds, k: clusteralgs.rbf_spectralcluster(ds, k, gamma)),
        "Scipy KNN": (lambda ds, k: clusteralgs.knn_spectralcluster(ds, k)),
        "IFGT FSC": (lambda ds, k: clusteralgs.fast_spectral_cluster_ifgt(ds, k, gamma)),
        "FAISS Exact": (lambda ds, k: clusteralgs.faiss_exact_spectral_cluster(ds, k)),
        "FAISS HNSW": (lambda ds, k: clusteralgs.faiss_hnsw_spectral_cluster(ds, k)),
        "FAISS IVF": (lambda ds, k: clusteralgs.faiss_ivf_spectral_cluster(ds, k)),
    }

    downsample_size = 20000

    # for num_vertices in [float('inf'), downsample_size]:
    for num_vertices in [downsample_size]:
        experimental_data = {}
        for alg_name, func in algorithms_to_compare.items():
            if num_vertices != downsample_size and alg_name == "Scipy RBF":
                # Ignore the full resolution image for SKLearn RBF
                continue
            experimental_data[alg_name] = []

            dataset = datasets.BSDSDataset(image_idx, max_vertices=num_vertices)
            print(f"Running {alg_name} on {dataset.num_data_points} pixels.")

            # Get the number of clusters for this dataset
            k = dataset.get_num_clusters()

            tstart = time()
            sc_labels = func(dataset, k)
            tend = time()
            duration = tend - tstart
            experimental_data[alg_name].append(ExperimentRunData(dataset, duration, sc_labels))
            print(f"{alg_name}: {duration:0.3f} seconds, {experimental_data[alg_name][-1].ari:0.3f} ARI")

        print("")

        filename = f"results/bsds/{image_idx}{'_downsampled' if num_vertices == downsample_size else ''}.pickle"
        with open(filename, 'wb') as fout:
            pickle.dump(experimental_data, fout)


def bsds_experiment_plot(image_idx, save=False):
    with open(f"results/bsds/{image_idx}.pickle", 'rb') as fin:
        experimental_data = pickle.load(fin)
    with open(f"results/bsds/{image_idx}_downsampled.pickle", 'rb') as fin:
        downsampled_data = pickle.load(fin)

    # Default to the downsampled data, otherwise use full-resolution
    data_dict = {}
    for alg_name, data in downsampled_data.items():
        data_dict[alg_name] = data
    for alg_name, data in experimental_data.items():
        data_dict[alg_name] = data
    del experimental_data
    del downsampled_data

    alg_to_file = {
        "IFGT FSC": "fast",
        "Scipy KNN": "knn",
        "Scipy RBF": "rbf",
        "FAISS Exact": "faiss_exact",
        "FAISS HNSW": "faiss_hnsw",
        "FAISS IVF": "faiss_ivf"
    }

    max_resolution = 0
    best_image_dataset = None
    for alg_name, data in data_dict.items():
        if len(data) > 0:
            data[-1].dataset.show_image(labels=data[-1].labels)

            # Save the current figure
            if save:
                plt.savefig(f"results/figures/{image_idx}_{alg_to_file[alg_name]}.png",
                            bbox_inches='tight')

            # Show the current figure
            print(f"Plotting segmentation from {alg_name}...")
            plt.show()

            if data[-1].dataset.num_data_points > max_resolution:
                max_resolution = data[-1].dataset.num_data_points
                best_image_dataset = data[-1].dataset

    print("Plotting original image...")
    best_image_dataset.show_image()
    if save:
        plt.savefig(f"results/figures/{image_idx}_orig.png",
                    bbox_inches='tight')
    plt.show()


def bsds_experiment_ari(image_idx):
    with open(f"results/bsds/{image_idx}_downsampled.pickle", 'rb') as fin:
        experimental_data = pickle.load(fin)

    aris = {}

    for alg_name, data in experimental_data.items():
        if len(data) > 0:
            # Create a new dataset object
            ds = datasets.BSDSDataset(image_idx,
                                      max_vertices=data[-1].dataset.max_vertices)

            # Compute the ARI
            ari = ds.ari(data[-1].labels)
            aris[alg_name] = ari

    return aris


def all_bsds_aris():
    images = [225022, 223060, 208078, 257098, 250047, 306051, 108004, 35049, 16068, 2018]
    images += [8068, 36046, 80090, 100007, 101027, 181021, 285022, 372019]
    images += [26031, 35008, 41004, 61060, 135069, 253036, 323016]

    ari_totals = {}

    for img_idx in images:
        print(f"Getting ARIs for image {img_idx}...")
        aris = bsds_experiment_ari(img_idx)

        for alg_name, ari in aris.items():
            print(f"{alg_name}: {ari}")

            if alg_name not in ari_totals:
                ari_totals[alg_name] = 0
            ari_totals[alg_name] += ari
        print()

    # Get the average ARIs
    print("Average ARIs")
    for alg_name, total_ari in ari_totals.items():
        print(f"{alg_name}: {total_ari / len(images)}")


def full_bsds_experiment():
    images = [225022, 223060, 208078, 257098, 250047, 306051, 108004, 35049, 16068, 2018]
    images += [8068, 36046, 80090, 100007, 101027, 181021, 285022, 372019]
    images += [26031, 35008, 41004, 61060, 135069, 253036, 323016]
    for image_idx in images:
        bsds_experiment(image_idx)


def parse_args():
    parser = argparse.ArgumentParser(description="Run experiments.")
    parser.add_argument('command', type=str, choices=['plot', 'run'])
    parser.add_argument('experiment', type=str,
                        choices=['moons', 'bsds', 'gap'])
    parser.add_argument('--id', required=False, help='ID of the bsds image to operate on')
    return parser.parse_args()


def main():
    args = parse_args()

    ########################
    # Two Moons
    ########################
    if args.experiment == 'moons':
        if args.command == 'run':
            two_moons_experiment()
        elif args.command == 'plot':
            two_moons_experiment_plot(save=True)

    ########################
    # BSDS
    ########################
    if args.experiment == 'bsds':
        if args.command == 'run':
            if not args.id:
                full_bsds_experiment()
            else:
                bsds_experiment(int(args.id))
        elif args.command == 'plot':
            all_bsds_aris()
            if not args.id:
                print("Must specify --id argument to plot BSDS image result.")
            else:
                bsds_experiment_plot(int(args.id))


if __name__ == "__main__":
    main()
