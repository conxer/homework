# 基于 Pix2Pix + GAN 的图像语义分割

本实验将 Pix2Pix 和 GAN 结合，实现了图像的语义分割。

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

- [My awesome model](pix2pix_model_epoch_800.pth) 

## Results

After 800 epochs were trained, the loss value of the training set reached 0.0543, and the loss value of the validation set reached 0.0608, and the predicted image is as follows:

### [Model performance]

| results    |  the training | the validation |
| ------------------ |---------------- | -------------- |
| loss value   |     0.0543         |      0.0608       |

### [The result predicted by the training set]
<img src="results/train_results.png" alt="train_results" width="800">

### [The result predicted by the validation set]
<img src="results/val_results.png" alt="val_results" width="800">
