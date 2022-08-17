import glob
import config
from matplotlib import pyplot as plt
from dataset import combine_masks, binary_from_polygon, load_image


def check_for_lung_overlap():
    targets = sorted(glob.glob(config.LABEL_DIR + "/*"))

    train_end = int(len(targets) * config.TRAIN_SIZE)
    val_end = train_end + int(len(targets) * config.VAL_SIZE)

    for img in targets[train_end:val_end]:
        print(img)
        combine_masks(*binary_from_polygon(img))


def show_sample():
    # Sample image as numpy array
    dicom_image = load_image(config.DATA_DIR + "/DO-2415411260-0610280953-7226048946-286786.dcm")

    # Sample mask as numpy array
    left, right = binary_from_polygon(config.LABEL_DIR + "/DO-2415411260-0610280953-7226048946-286786.json")
    mask = combine_masks(left, right)

    fig = plt.figure()

    fig.add_subplot(1, 2, 1)
    plt.imshow(dicom_image, cmap=plt.cm.bone)

    fig.add_subplot(1, 2, 2)
    plt.imshow(mask, interpolation=None)

    plt.show()

