import glob
import os
import numpy as np
import config
from tqdm import tqdm
from skimage import transform
from dataset import load_image, combine_masks, binary_from_polygon

__author__ = "Christian Grashei"


def preprocess_data():
    """
    Preprocessing of the X-Ray images and JSON ground truth for training the U-Net model. The DICOM images are loaded,
    converted to a numpy array, resized and scaled to values between 0. and 1.
    The JSON ground truth is loaded, converted to a numpy array representing the different classes by integers
    and resized.
    The created numpy arrays are stored on disk at the specified path in the config.
    """
    input_files = sorted(glob.glob(config.DATA_DIR + "/*"))
    target_files = sorted(glob.glob(config.LABEL_DIR + "/*"))

    with tqdm(input_files, unit="img", desc="Preprocessing DICOMs") as pbar:
        for file in pbar:
            filename = os.path.basename(file)
            filename = os.path.splitext(filename)[0] + ".npy"
            img = load_image(file)
            img = img.astype(np.float64)
            img = transform.resize(img, (config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH), preserve_range=True)
            img = img / np.amax(img)
            np.save(config.PREPROCESSING_PATH + "/images/" + filename, img)

    with tqdm(target_files, unit="json", desc="Preprocessing JSONs") as pbar:
        for file in pbar:
            filename = os.path.basename(file)
            filename = os.path.splitext(filename)[0] + ".npy"
            mask = combine_masks(*binary_from_polygon(file))
            mask = transform.resize(mask, (config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH), order=0,
                                    preserve_range=True,
                                    anti_aliasing=False)
            mask = np.rint(mask).astype(np.long)
            np.save(config.PREPROCESSING_PATH + "/masks/" + filename, mask)


if __name__ == "__main__":
    preprocess_data()
