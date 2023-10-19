"""
This module provides functions for interacting with different datasets.
"""
from typing import Optional, List
import keras.datasets.mnist
import numpy as np
import scipy.io
import skimage.transform
import skimage.measure
import skimage.filters
import sklearn.metrics
from sklearn import datasets
from sklearn import random_projection
import random
import math
import matplotlib
import matplotlib.pyplot as plt
import os
from matplotlib import image


class Dataset(object):
    """
    This base class represents a dataset, for clustering. A dataset might consist of some combination of:
      - raw numerical data
      - a ground truth clustering
    """

    def __init__(self, raw_data=None):
        """
        Intiialise the dataset, optionally specifying a data file.
        """
        self.raw_data = raw_data
        self.gt_labels: Optional[List[int]] = None
        self.load_data()
        self.num_data_points = self.raw_data.shape[0] if self.raw_data is not None else 0
        self.data_dimension = self.raw_data.shape[1] if self.raw_data is not None else 0

    def load_data(self):
        """
        Load the data for this dataset. The implementation may differ significantly
        from dataset to dataset.
        """
        pass

    def plot(self, labels):
        """
        Plot the dataset with the given labels.
        """
        if self.data_dimension > 3:
            print("Cannot plot data with dimensionality above 3.")
            return

        if self.data_dimension == 2:
            plt.scatter(self.raw_data[:, 0], self.raw_data[:, 1], c=labels, marker='.')
        else:
            # Create the figure
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(self.raw_data[:, 0], self.raw_data[:, 1], self.raw_data[:, 2], c=labels, marker='.')
        plt.show()

    def ari(self, labels):
        """
        Compute the Adjusted Rand Index of the given candidate labels.
        """
        if self.gt_labels is not None:
            return sklearn.metrics.adjusted_rand_score(self.gt_labels, labels)
        else:
            return 0

    def normalise(self):
        """
        Normalise the data to lie between 0 and 1.
        """
        # Get the maximum and minimum values in each dimension
        self.data_dimension = self.raw_data.shape[1] if self.raw_data is not None else 0
        dimension_max_vals = [float('-inf')] * self.data_dimension
        dimension_min_vals = [float('inf')] * self.data_dimension
        for d in range(self.data_dimension):
            for point in self.raw_data:
                if point[d] > dimension_max_vals[d]:
                    dimension_max_vals[d] = point[d]
                if point[d] < dimension_min_vals[d]:
                    dimension_min_vals[d] = point[d]

        # Find the amount we need to scale down by
        scale_factor = 0
        for d in range(self.data_dimension):
            this_range = dimension_max_vals[d] - dimension_min_vals[d]
            if this_range > scale_factor:
                scale_factor = this_range

        # Normalise all of the data
        for i, point in enumerate(self.raw_data):
            for d in range(self.data_dimension):
                self.raw_data[i, d] = (point[d] - dimension_min_vals[d]) / scale_factor

    def __repr__(self):
        return self.__str__()


class TwoMoonsDataset(Dataset):

    def __init__(self, n=1000):
        """
        Create an instance of the two moons dataset with the specified number of
        data points.
        """
        self.n = n
        super(TwoMoonsDataset, self).__init__()

    def load_data(self):
        self.raw_data, self.gt_labels = datasets.make_moons(n_samples=self.n, noise=0.05)
        self.data_dimension = 2
        self.normalise()

    def __str__(self):
        return f"twoMoonsDataset(n={self.num_data_points})"


class BSDSDataset(Dataset):

    def __init__(self, img_idx, max_vertices=None,
                 data_directory=os.path.join(os.getcwd(), os.pardir, "data/bsds/BSR/BSDS500/data/"),
                 blur_variance=1):
        """Construct a dataset from a single image in the BSDS dataset.

        We will construct a graph from the image based on the gaussian radial
        basis function.

        :param img_idx: The number of the image in the dataset.
        :param max_vertices: The maximum number of pixels in the image, will be
                             downsampled to match.
        :param blur_variance: The variance of the gaussian blur applied to the downsampled image
        :param data_directory: The base directory containing the dataset images.
        """
        self.img_idx = img_idx
        self.image_filename, self.gt_filename = self.get_filenames(data_directory,
                                                                   img_idx)
        self.original_image_dimensions = []
        self.downsampled_image_dimensions = []
        self.blur_variance = blur_variance
        self.original_image = None
        self.downsampled_image = None
        self.max_vertices = max_vertices
        self.downsample_factor = 1
        self.gt_labels = None
        super(BSDSDataset, self).__init__()

    @staticmethod
    def get_filenames(base_directory: str, img_id: int):
        """
        Get the image and ground truth filenames for the given BSDS image.
        """
        image_filename = f"{img_id}.jpg"
        ground_truth_filename = f"{img_id}.mat"

        # Figure out whether it is test or training data
        images_directory = os.path.join(base_directory, "images/test/")
        if image_filename in os.listdir(images_directory):
            ground_truth_directory = os.path.join(base_directory, "groundTruth/test/")
        else:
            images_directory = os.path.join(base_directory, "images/train/")
            ground_truth_directory = os.path.join(base_directory, "groundTruth/train/")

        # Make sure that both files exist before returning them
        img_path = os.path.join(images_directory, image_filename)
        gt_path = os.path.join(ground_truth_directory, ground_truth_filename)
        if not os.path.isfile(img_path) or not os.path.isfile(gt_path):
            raise Exception("BSDS file not found")

        return img_path, gt_path

    def get_num_clusters(self):
        gt_data = scipy.io.loadmat(self.gt_filename)
        num_gt_segs = gt_data["groundTruth"].shape[1]

        num_segments = []
        for i in range(num_gt_segs):
            this_segmentation = gt_data["groundTruth"][0, i][0][0][0]
            this_num_segments = np.max(this_segmentation)
            num_segments.append(this_num_segments)

        # Get the median number of segments
        gt_num_segments = max(2, int(np.median(num_segments)))

        # Set the ground truth clustering to the one closest to the GT number
        # of segments
        dist = float('inf')
        for i in range(num_gt_segs):
            this_segmentation = gt_data["groundTruth"][0, i][0][0][0]
            this_num_segments = np.max(this_segmentation)
            if abs(this_num_segments - gt_num_segments) < dist:
                dist = abs(this_num_segments - gt_num_segments)
                self.set_gt_labels(this_segmentation)

        # Return the median number of segments, and at least 2.
        return gt_num_segments

    def set_gt_labels(self, segmentation):
        self.gt_labels = list(np.ndarray.flatten(segmentation))

    def ari(self, labels) -> float:
        if self.max_vertices is not None:
            # Scale up the labels
            img = np.reshape(labels, self.downsampled_image_dimensions)
            upscale_img = skimage.transform.rescale(img,
                                                    1 / self.downsample_factor,
                                                    preserve_range=True)
            upscale_img = upscale_img[:self.original_image_dimensions[0],
                                      :self.original_image_dimensions[1]]
            assert upscale_img.shape == self.original_image_dimensions

            # Get the labels list
            new_labels = list(np.ndarray.flatten(upscale_img))
            return sklearn.metrics.rand_score(self.gt_labels, list(new_labels))
        else:
            return sklearn.metrics.rand_score(self.gt_labels, list(labels))

    def load_data(self):
        """
        Load the dataset from the image. Each pixel in the image is a data point. Each data point will have 5
        dimensions, namely the normalised 'rgb' values and the (x, y) coordinates of the pixel in the image.

        To reformat the data to be a manageable size, we downsample by a factor of 3.

        :return:
        """
        img = image.imread(self.image_filename)
        self.original_image_dimensions = (img.shape[0], img.shape[1])
        self.original_image = img / 255

        # Compute the downsample factor if needed
        orig_num_pixels = img.shape[0] * img.shape[1]
        self.downsample_factor = 1
        if self.max_vertices is not None:
            self.downsample_factor = min(1,
                math.sqrt(self.max_vertices / orig_num_pixels))

        # Do the downsampling here
        img = skimage.transform.rescale(img, self.downsample_factor,
                                        preserve_range=True,
                                        channel_axis=2)
        self.downsampled_image_dimensions = (img.shape[0], img.shape[1])

        # Blur the image slightly
        if self.blur_variance > 0:
            img = skimage.filters.gaussian(img, sigma=self.blur_variance)

        self.downsampled_image = img / 255

        # Extract the data points from the image
        self.raw_data = []
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                self.raw_data.append([img[x, y, 0] / 255,
                                      img[x, y, 1] / 255,
                                      img[x, y, 2] / 255,
                                      x / max(img.shape[0], img.shape[1], 255),
                                      y / max(img.shape[0], img.shape[1], 255)])
        self.raw_data = np.array(self.raw_data)

        # Get the ground truth labels - it is implicitly populated by the
        # get_num_clusters method.
        k = self.get_num_clusters()

    def show_image(self, labels=None):
        """
        Display the image.
        """
        if labels is None:
            plt.imshow(self.downsampled_image)
            plt.xticks([])
            plt.yticks([])
        else:
            # Reshape the labels to match the downsampled image dimensions
            label_img = np.reshape(labels, self.downsampled_image_dimensions)
            plt.imshow(label_img,
                       cmap=matplotlib.colormaps['jet'],
                       interpolation='none')
            plt.xticks([])
            plt.yticks([])

    def __str__(self):
        return f"bsds({self.img_idx})"
