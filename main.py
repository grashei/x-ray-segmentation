import glob
import torch
import segmentation_models_pytorch as smp
from matplotlib import pyplot as plt
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils import data
from torchmetrics import JaccardIndex, F1Score
from sklearn.metrics import jaccard_score, f1_score
from tqdm import tqdm

import config
from dataset import SegmentationDataSet
from dataset import binary_from_polygon
from dataset import combine_masks
from dataset import load_image


def train(input_files, target_files, train_length, epochs=1, batch_size=32, lr=0.001, encoder_name="mobilenet_v2", activation="sigmoid"):
    training_dataset = SegmentationDataSet(input_files[0:train_length], target_files[0:train_length], transform=None)

    training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size,
                                                      shuffle=True, num_workers=4)

    model = smp.Unet(
        encoder_name=encoder_name,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=1,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=3,  # model output channels (number of classes in your dataset),
        activation=activation
    )

    criterion = nn.CrossEntropyLoss()
    # criterion_alt = smp.losses.JaccardLoss(mode="multiclass")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = ExponentialLR(optimizer, gamma=0.75)

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

        lr_scheduler.step()

    print("Finished Training")

    return model


def test(model, input_files, target_files, train_length, batch_size=32):
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
            sm_running_iou_score += smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro")

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
    plt.imshow(input_test[0][0], cmap=plt.cm.bone)

    fig.add_subplot(1, 3, 2)
    plt.imshow(labels[0].long(), interpolation="nearest")

    fig.add_subplot(1, 3, 3)
    plt.imshow(output_mask[0], interpolation="nearest")

    plt.show()



if __name__ == "__main__":
    # Sample mask as numpy array
    left, right = binary_from_polygon(config.LABEL_DIR + "/DO-1000449909-0908270917-4526879995-901387.json")
    combine_masks(left, right)

    # Sample image as numpy array
    dicom_image = load_image(config.DATA_DIR + "/DO-1000449909-0908270917-4526879995-901387.dcm")

    # Get images from directory
    inputs = sorted(glob.glob(config.DATA_DIR + "/*"))
    # Get label files from directory
    targets = sorted(glob.glob(config.LABEL_DIR + "/*"))

    # Length of input files and target files must be the same
    assert len(inputs) == len(targets)

    train_length = int(len(inputs) * 0.98)

    model = train(inputs, targets, train_length, epochs=config.EPOCHS, batch_size=config.BATCH_SIZE)
    test(model, inputs, targets, train_length, batch_size=config.BATCH_SIZE)

    show_image_with_segmentation_masks(model, inputs, targets, train_length)
