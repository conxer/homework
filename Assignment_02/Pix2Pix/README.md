# Implement Pix2Pix with Fully Convolutional Layers

This repository is the official implementation of [Implement Pix2Pix with Fully Convolutional Layers](https://arxiv.org/abs/1411.4038). 

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Datasets

To have the datasets:, run:

```data
bash download_facades_dataset.sh 
```

## Training

To train the model(s) in the paper, run this command:

```train
python train.py 
```

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) 

## Results

After 800 epochs were trained, the loss value of the training set reached 0.0543, and the loss value of the validation set reached 0.0608, and the predicted image is as follows:

### [Model performance]

| results    |  the training | the validation |
| ------------------ |---------------- | -------------- |
| loss value   |     0.0543         |      0.0608       |

### [The result predicted by the training set]

### [The result predicted by the validation set]

