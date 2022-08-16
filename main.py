import glob
import torch
import numpy as np
import segmentation_models_pytorch as smp
import argparse
from matplotlib import pyplot as plt
from torch import optim
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from torch.utils import data
from torchmetrics import JaccardIndex, F1Score
from torchsummary import summary
from sklearn.metrics import jaccard_score, f1_score
from tqdm import tqdm

import config
from dataset import SegmentationDataSet
from dataset import binary_from_polygon
from dataset import combine_masks
from dataset import load_image


def train(input_files, target_files, train_length, batch_size=32, epochs=1, lr=0.001, encoder_name="mobilenet_v2",
          encoder_weights="imagenet", activation="sigmoid"):
    training_dataset = SegmentationDataSet(input_files[0:train_length], target_files[0:train_length], transform=None)

    training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size,
                                                      shuffle=True, num_workers=4, persistent_workers=True)

    model = smp.Unet(
        encoder_name=encoder_name,  # encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights=encoder_weights,  # use `imagenet` pre-trained weights
        in_channels=1,
        classes=3,
        activation=activation
    )

    criterion = torch.nn.CrossEntropyLoss()
    #criterion = smp.losses.JaccardLoss(mode="multiclass")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    #lr_scheduler = ExponentialLR(optimizer, gamma=0.75)
    lr_scheduler = ReduceLROnPlateau(optimizer, 'min', patience=4, threshold=0.006, verbose=True)

    # Training for EPOCHS
    for epoch in range(epochs):
        with tqdm(training_dataloader, unit="batch", desc=f"Epoch {epoch + 1}") as pbar:
            for data in pbar:
                running_loss = 0.0
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

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
               f"{config.MODEL_PATH}/{encoder_name}_{batch_size}_{epochs}_{lr}_{activation}.pth")
    return model


def test(model, input_files, target_files, train_length, batch_size=32):
    # Worse performance using eval??
    model.eval()
    summary(model, (1, config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH))

    test_dataset = SegmentationDataSet(input_files[train_length:], target_files[train_length:], transform=None)

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                                  shuffle=False, num_workers=4)

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

        for data in test_dataloader:
            inputs, labels = data
            # inputs = inputs.float()
            # labels = labels.long()
            outputs = model(inputs)
            outputs = torch.argmax(outputs, dim=1)
            total_batches += 1

            # Calculate mIoU with Segmentation models library
            tp, fp, fn, tn = smp.metrics.get_stats(outputs, labels, mode='multiclass', num_classes=3)
            sm_running_iou_score += smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

            # Calculate mIoU with torchmetrics library
            tm_running_iou_score += jaccard(outputs, labels)

            # Calculate mIoU with scikit-learn
            for label_sample, output_sample in zip(labels, outputs):
                sklearn_running_iou_score += \
                    jaccard_score(label_sample.numpy().flatten(), output_sample.numpy().flatten(), average="macro")
            sklearn_running_iou_score /= labels.shape[0]

            # Calculate F-Score with Segmentation Models library
            sm_running_f1_score += smp.metrics.f1_score(tp, fp, fn, tn, reduction="macro")

            # Calculate F-Score with torchmetrics library
            tm_running_f1_score += f1(outputs, labels)

            # Calculate F-Score with scikit-learn library
            for label_sample, output_sample in zip(labels, outputs):
                sklearn_running_f1_score += \
                    f1_score(label_sample.numpy().flatten(), output_sample.numpy().flatten(), average="macro")
            sklearn_running_f1_score /= labels.shape[0]

    print(f"Segmentation models mIoU: {sm_running_iou_score / total_batches}")
    print(f"torchmetrics mIoU: {tm_running_iou_score / total_batches}")
    print(f"scikit-learn mIoU: {sklearn_running_iou_score / total_batches}")

    print(f"Segmentation models F1-Score: {sm_running_f1_score / total_batches}")
    print(f"torchmetrics F1-Score: {tm_running_f1_score / total_batches}")
    print(f"scikit-learn F1-Score: {sklearn_running_f1_score / total_batches}")


def show_image_with_segmentation_masks(model, input_files, target_files, train_length):
    test_dataset = SegmentationDataSet(input_files[train_length:], target_files[train_length:], transform=None)

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4,
                                                  shuffle=False, num_workers=4)

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


def show_sample():
    # Sample image as numpy array
    dicom_image = load_image(config.DATA_DIR + "/DO-1000449909-0908270917-4526879995-901387.dcm")

    # Sample mask as numpy array
    left, right = binary_from_polygon(config.LABEL_DIR + "/DO-1000449909-0908270917-4526879995-901387.json")
    mask = combine_masks(left, right)

    fig = plt.figure()

    fig.add_subplot(1, 2, 1)
    plt.imshow(dicom_image, cmap=plt.cm.bone)

    fig.add_subplot(1, 2, 2)
    plt.imshow(mask, interpolation=None)

    plt.show()


def test_samples():
    # Get images from directory
    samples = sorted(glob.glob(config.SAMPLES_DIR + "/*"))
    targets = sorted(glob.glob(config.LABEL_DIR + "/*"))

    samples_dataset = SegmentationDataSet(samples, targets[:len(samples)], transform=None)
    samples_dataloader = torch.utils.data.DataLoader(samples_dataset, batch_size=1, shuffle=False, num_workers=4)

    model = torch.load("/Users/christiangrashei/Desktop/Siemens/Models/efficientnet-b0_4_30_0.001_sigmoid.pth")
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
            #plt.imshow(output_mask[0], interpolation=None)
            plt.imshow(sample[0][0], cmap=plt.cm.bone)

            plt.show()


def main(args):

    # Get images from directory
    inputs = sorted(glob.glob(config.DATA_DIR + "/*"))
    # Get label files from directory
    targets = sorted(glob.glob(config.LABEL_DIR + "/*"))

    # Length of input files and target files must be the same
    assert len(inputs) == len(targets)

    train_length = int(len(inputs) * config.TRAIN_SIZE)

    if not args.model_path:
        model = train(inputs,
                      targets,
                      train_length,
                      epochs=args.epochs,
                      batch_size=args.batch_size,
                      lr=args.lr,
                      encoder_name=args.encoder,
                      encoder_weights="imagenet" if args.weights == "y" else None,
                      activation=args.activation)

        test(model, inputs, targets, train_length, batch_size=args.batch_size)
    else:
        model = torch.load(args.model_path)
        test(model, inputs, targets, train_length, batch_size=args.batch_size)

    show_image_with_segmentation_masks(model, inputs, targets, train_length)


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

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
    #main(parsed_args)
    test_samples()

