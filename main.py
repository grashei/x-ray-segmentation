import glob
import numpy
import torch
import numpy as np
import segmentation_models_pytorch as smp
import argparse
from matplotlib import pyplot as plt
from torch import optim
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from torch.utils import data
from torchmetrics import JaccardIndex, F1Score
from sklearn.metrics import jaccard_score, f1_score
from tqdm import tqdm

import config
from dataset import SegmentationDataSet, PerfSegmentationDataSet


def get_perf_dataloader(batch_size, training=True):
    # Get images from directory
    input_files = sorted(glob.glob(config.PREPROCESSING_PATH + "/images/" + "/*"))
    # Get label files from directory
    target_files = sorted(glob.glob(config.PREPROCESSING_PATH + "/masks/" + "/*"))

    # Length of input files and target files must be the same
    assert len(input_files) == len(target_files)

    train_end = int(len(input_files) * config.TRAIN_SIZE)

    if training:
        dataset = PerfSegmentationDataSet(input_files[:train_end], target_files[:train_end])
    else:
        val_end = train_end + int(len(input_files) * config.VAL_SIZE)
        dataset = SegmentationDataSet(input_files[train_end:val_end], target_files[train_end:val_end])

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=8,
                                             pin_memory=True,
                                             prefetch_factor=10)

    return dataloader


def get_dataloader(batch_size, training=True):
    """
    Creates a dataloader for the training or validation partition of the dataset. The size of the training and
    Validation partition can be specified in the config file

    Parameters
    ----------
    training: If true the dataloader for the training partition is created, otherwise for the validation partition
    batch_size: The batch size the dataloader should have

    Returns
    -------
    dataloader : The dataloader for the requested dataset
    """
    # Get images from directory
    input_files = sorted(glob.glob(config.DATA_DIR + "/*"))
    # Get label files from directory
    target_files = sorted(glob.glob(config.LABEL_DIR + "/*"))

    # Length of input files and target files must be the same
    assert len(input_files) == len(target_files)

    train_end = int(len(input_files) * config.TRAIN_SIZE)

    if training:
        dataset = PerfSegmentationDataSet(input_files[:train_end], target_files[:train_end])
    else:
        val_end = train_end + int(len(input_files) * config.VAL_SIZE)
        dataset = SegmentationDataSet(input_files[train_end:val_end], target_files[train_end:val_end])

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=8,
                                             pin_memory=True,
                                             prefetch_factor=10)

    return dataloader


def train(batch_size=32, epochs=1, lr=0.001, encoder_name="mobilenet_v2",
          encoder_weights="imagenet", activation="sigmoid"):
    #training_dataloader = get_dataloader(batch_size, training=True)
    training_dataloader = get_perf_dataloader(batch_size, training=True)

    model = smp.Unet(
        encoder_name=encoder_name,  # encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights=encoder_weights,  # use `imagenet` pre-trained weights
        in_channels=1,
        classes=3,
        activation=activation
    )

    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = ReduceLROnPlateau(optimizer, 'min', patience=4, threshold=0.006, verbose=True)

    # Training for EPOCHS
    for epoch in range(epochs):
        with tqdm(training_dataloader, unit="batch", desc=f"Epoch {epoch + 1}") as pbar:
            for data in pbar:
                running_loss = 0.0
                inputs, labels = data[0].to(device=device, non_blocking=True), data[1].to(device=device,
                                                                                          non_blocking=True)

                # zero the parameter gradients
                optimizer.zero_grad(set_to_none=True)

                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # update statistics
                running_loss += loss.item()
                pbar.set_postfix(loss=running_loss)

        lr_scheduler.step(running_loss)

    print("Finished Training")

    torch.save(model,
               f"{config.MODEL_DIR}/{encoder_name}_{batch_size}_{epochs}_{lr}_{activation}.pth")
    return model


def test(model, batch_size=32):
    # Worse performance using eval??
    model.eval()
    model.to("cpu")

    #val_dataloader = get_dataloader(batch_size, training=False)
    val_dataloader = get_perf_dataloader(batch_size, training=False)

    jaccard = JaccardIndex(num_classes=3, average="macro")
    f1 = F1Score(num_classes=3, average="macro", mdmc_average="global")

    with torch.no_grad():
        total_batches = 0
        sm_running_iou_score = 0
        tm_running_iou_score = 0
        sklearn_running_iou_score = 0

        sm_running_f1_score = 0
        tm_running_f1_score = 0
        sklearn_running_f1_score = 0

        for data in val_dataloader:
            inputs, labels = data
            outputs = model(inputs)
            outputs = torch.argmax(outputs, dim=1)
            total_batches += 1

            # Calculate mIoU with Segmentation models library
            tp, fp, fn, tn = smp.metrics.get_stats(outputs, labels, mode='multiclass', num_classes=3)
            sm_running_iou_score += smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro")

            # Calculate mIoU with torchmetrics library
            tm_running_iou_score += jaccard(outputs, labels)

            # Calculate mIoU with scikit-learn
            sklearn_per_class_average_iou = numpy.zeros(3)
            for label_sample, output_sample in zip(labels, outputs):
                sklearn_per_class_average_iou += jaccard_score(label_sample.numpy().flatten(),
                                                               output_sample.numpy().flatten(), average=None)
            sklearn_per_class_average_iou /= labels.shape[0]
            sklearn_running_iou_score += np.average(sklearn_per_class_average_iou)

            # Calculate F-Score with Segmentation Models library
            sm_running_f1_score += smp.metrics.f1_score(tp, fp, fn, tn, reduction="macro")

            # Calculate F-Score with torchmetrics library
            tm_running_f1_score += f1(outputs, labels)

            # Calculate F-Score with scikit-learn library
            sklearn_per_class_average_f1 = numpy.zeros(3)
            for label_sample, output_sample in zip(labels, outputs):
                sklearn_per_class_average_f1 += f1_score(label_sample.numpy().flatten(),
                                                         output_sample.numpy().flatten(), average=None)
            sklearn_per_class_average_f1 /= labels.shape[0]
            sklearn_running_f1_score += np.average(sklearn_per_class_average_f1)

    print(f"Segmentation models mIoU: {sm_running_iou_score / total_batches}")
    print(f"torchmetrics mIoU: {tm_running_iou_score / total_batches}")
    print(f"scikit-learn mIoU: {sklearn_running_iou_score / total_batches}")

    print(f"Segmentation models F1-Score: {sm_running_f1_score / total_batches}")
    print(f"torchmetrics F1-Score: {tm_running_f1_score / total_batches}")
    print(f"scikit-learn F1-Score: {sklearn_running_f1_score / total_batches}")


def show_image_with_segmentation_masks(model):
    model.eval()
    model.to("cpu")

    test_dataloader = get_dataloader(1, training=False)

    dataiter = iter(test_dataloader)
    input_test, labels = dataiter.next()
    outputs = model(input_test)
    output_mask = torch.argmax(outputs, dim=1)

    fig = plt.figure()

    fig.add_subplot(1, 3, 1)
    plt.imshow(input_test[1][0], cmap=plt.cm.bone)

    fig.add_subplot(1, 3, 2)
    plt.imshow(labels[1].long(), interpolation=None)

    fig.add_subplot(1, 3, 3)
    plt.imshow(output_mask[1], interpolation=None)

    plt.show()


def test_samples(model_path):
    # Get images from directory
    samples = sorted(glob.glob(config.SAMPLES_DIR + "/*"))
    targets = sorted(glob.glob(config.LABEL_DIR + "/*"))

    samples_dataset = SegmentationDataSet(samples, targets[:len(samples)])
    samples_dataloader = torch.utils.data.DataLoader(samples_dataset, batch_size=1, shuffle=False)

    model = torch.load(model_path)
    model.eval()

    with torch.no_grad():
        dataiter = iter(samples_dataloader)
        for sample, _ in dataiter:
            outputs = model(sample)
            output_mask = torch.argmax(outputs, dim=1)

            fig = plt.figure(figsize=(20, 10))

            fig.add_subplot(1, 2, 1)
            plt.imshow(sample[0][0], cmap=plt.cm.bone)
            plt.imshow(output_mask[0], interpolation=None, alpha=0.5)

            fig.add_subplot(1, 2, 2)
            plt.imshow(sample[0][0], cmap=plt.cm.bone)

            plt.show()


def main(args):
    if not args.model_path:
        model = train(epochs=args.epochs,
                      batch_size=args.batch_size,
                      lr=args.lr,
                      encoder_name=args.encoder,
                      encoder_weights="imagenet" if args.weights == "y" else None,
                      activation=args.activation)

        test(model, batch_size=args.batch_size)
    else:
        model = torch.load(args.model_path)
        test(model, batch_size=args.batch_size)

    # show_image_with_segmentation_masks(model)


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()

    parser.add_argument("-a", "--activation",
                        dest="activation",
                        type=str,
                        choices=["sigmoid", "softmax", "logsoftmax", "tanh", "identity"],
                        required=False,
                        default="sigmoid",
                        help="Activation function to use")

    parser.add_argument("-m", "--model",
                        dest="encoder",
                        type=str,
                        choices=["densenet121", "efficientnet-b0", "mobilenet_v2", "resnet18"],
                        required=False,
                        default="mobilenet_v2",
                        help="Encoder to use")

    parser.add_argument("-bs", "--batch_size",
                        dest="batch_size",
                        type=int,
                        required=False,
                        default=config.BATCH_SIZE,
                        help="Batch size")

    parser.add_argument("-e", "--epochs",
                        dest="epochs",
                        type=int,
                        required=False,
                        default=config.EPOCHS,
                        help="Number of training epochs")

    parser.add_argument("-lr", "--learning_rate",
                        dest="lr",
                        type=float,
                        required=False,
                        default=config.LEARNING_RATE,
                        help="Learning rate to use")

    parser.add_argument("-s", "--size",
                        dest="size",
                        type=int,
                        required=False,
                        default=config.INPUT_IMAGE_WIDTH
                        if config.INPUT_IMAGE_WIDTH < config.INPUT_IMAGE_HEIGHT
                        else config.INPUT_IMAGE_HEIGHT,
                        help="Image resizing size")

    parser.add_argument("-w", "--weights",
                        dest="weights",
                        type=str,
                        choices=["y", "n"],
                        required=False,
                        default="y",
                        help="Use encoder weights pretrained on imagenet")

    parser.add_argument("-mp", "--model_path",
                        dest="model_path",
                        type=str,
                        required=False,
                        default=None,
                        help="Test model from specified location")

    parsed_args = parser.parse_args()
    main(parsed_args)
