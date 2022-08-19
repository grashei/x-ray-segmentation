import torch
import pydicom
import json
import numpy as np
from scipy import ndimage
from torch.utils import data
from skimage.draw import polygon
from skimage import transform

import config

__author__ = "Regine Büter, Christian Grashei"


def binary_from_polygon(path_to_json):
    """Create a binary segmentation mask from a json file which contains a polygon trace for each lung.

    Parameters
    ----------
    path_to_json: Path to the json file containing the polygon trace which should be translated to a binary mask.

    Returns
    -------
    left : The binary segmentation mask for the left lung
    right : The binary segmentation mask for the right lung
    """

    f = open(path_to_json)
    json_data = json.load(f)
    f.close()
    width = json_data["image"]["width"]
    height = json_data["image"]["height"]
    tmp = 0
    left = np.zeros((height, width))
    right = np.zeros((height, width))
    for i in range(0, len(json_data["annotations"])):
        if json_data["annotations"][i]["name"] == "lung boundary":
            polygon_trace = json_data["annotations"][i]["frames"]["0"]["polygon"]["path"]
            poly_xy = polygon_boundary(polygon_trace)
            rr, cc = polygon(poly_xy[0], poly_xy[1], left.shape)
            left[rr, cc] = 1
            tmp = i + 1
            break
    for j in range(tmp, len(json_data["annotations"])):
        if json_data["annotations"][j]["name"] == "lung boundary":
            polygon_trace = json_data["annotations"][j]["frames"]["0"]["polygon"]["path"]
            poly_xy = polygon_boundary(polygon_trace)
            rr, cc = polygon(poly_xy[0], poly_xy[1], left.shape)
            right[rr, cc] = 1
            break

    # Check whether right is actually the right lung and left the left lung by using center of mass
    center_left = ndimage.center_of_mass(left)
    center_right = ndimage.center_of_mass(right)
    if center_left[1] < center_right[1]:
        swap = left
        left = right
        right = swap
    return left, right


def polygon_boundary(polygon_trace):
    """Creates a numpy array containing the coordinates from a polygon trace.

    Parameters
    ----------
    polygon_trace: The polygon trace

    Returns
    -------
    trace : float or ndarray of shape (2, n_points), dtype=np.float64
            A numpy array containing the points of the polygon trace
    """
    x = []
    y = []
    for point in polygon_trace:
        x.append(float(point["x"]))
        y.append(point["y"])
    return np.array([y, x])


def load_image(path):
    """Reads a dicom file and converts the image to a pixel array.

    Parameters
    ----------
    path: Path of the dicom file

    Returns
    -------
    dcm_img : Pixel array of the image
    """
    ds = pydicom.dcmread(path)
    if ds[0x0028, 0x0004].value == 'MONOCHROME1':
        dcm_img = np.max(ds.pixel_array) - ds.pixel_array
    else:
        dcm_img = ds.pixel_array
    return dcm_img


def combine_masks(left_lung, right_lung):
    """Create a combined mask of the segmentation mask for the left and right lung by assigning label 0
    to the background, label 1 to the right lung and label 2 to the left lung.

    Parameters
    ----------
    left_lung : (M, N) ndarray
        Mask for the left lung.
    right_lung : (M, N) ndarray
        Mask for the right lung.
    Returns
    -------
    combined : Combined segmentation mask
    """

    left_lung = left_lung * 2
    combined = right_lung + left_lung

    # Is there an overlap between left and right lung? If yes, the combination would not be correct.
    assert np.amax(combined) <= 2

    return combined


class SegmentationDataSet(data.Dataset):
    """
    This class loads and prepares the images and segmentation mask and represents a dataset of them. Preprocessing is
    done on loading of each sample
    """

    def __init__(self, inputs: list, targets: list):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index: int):
        # Select the sample
        input_id = self.inputs[index]
        target_id = self.targets[index]

        # Load input and target
        x = load_image(input_id)
        y = combine_masks(*binary_from_polygon(target_id))

        x = x.astype(np.float64)
        x = transform.resize(x, (config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH), preserve_range=True)
        y = transform.resize(y, (config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH), order=0, preserve_range=True,
                             anti_aliasing=False)
        y = np.rint(y).astype(np.long)
        x = x / np.amax(x)

        x = np.expand_dims(x, axis=0)

        # Typecasting
        x, y = torch.from_numpy(x).float(), torch.from_numpy(y).long()

        return x, y


class PerfSegmentationDataSet(data.Dataset):
    """
    This class loads and prepares the images and segmentation mask and represents a dataset of them. Data needs to be
    preprocessed in advance (faster).
    """

    def __init__(self, inputs: list, targets: list):
        self.inputs = []
        self.targets = []

        for img in inputs:
            self.inputs.append(np.load(img))

        for mask in targets:
            self.targets.append(np.load(mask))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index: int):
        x = self.inputs[index]
        y = self.targets[index]
        x = np.expand_dims(x, axis=0)

        # Typecasting
        x, y = torch.from_numpy(x).float(), torch.from_numpy(y).long()

        return x, y
