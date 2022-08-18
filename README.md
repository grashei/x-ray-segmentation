# Segmentation of left and right lung in X-ray images

Before training the network the DICOM images and JSON groundtruth must be preprocessed. For this purpose the `preprocessing.py` can be used. For training the following CLI can be used:
```
optional arguments:
  -h, --help            show this help message and exit
  -a {sigmoid,softmax,logsoftmax,tanh,identity}, --activation {sigmoid,softmax,logsoftmax,tanh,identity}
                        Activation function to use
  -m {densenet121,efficientnet-b0,mobilenet_v2,resnet18}, --model {densenet121,efficientnet-b0,mobilenet_v2,resnet18}
                        Encoder to use
  -bs BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch size
  -e EPOCHS, --epochs EPOCHS
                        Number of training epochs
  -lr LR, --learning_rate LR
                        Learning rate to use
  -s SIZE, --size SIZE  Image resizing size
  -w {y,n}, --weights {y,n}
                        Use encoder weights pretrained on imagenet
  -mp MODEL_PATH, --model_path MODEL_PATH
                        Test model from specified location
```

For testing a model the `-mp` argument can be used to specify the model location