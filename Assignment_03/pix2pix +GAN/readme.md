# Image Semantic Segmentation Based on Pix2Pix + GAN
This experiment combines Pix2Pix and GAN to achieve image semantic segmentation.

## Explanation of Hyperparameters
The hyperparameters used for training are as follows:

### Learning Rate:
The learning rate of the Generator: 0.002
The learning rate of the Discriminator: 0.001

### Optimizer:
The Adam optimizer, betas = (0.5, 0.999), weight_decay = 1e - 5

### Learning Rate Scheduler:
StepLR, every 100 epochs, the learning rate decays to half of the original

### Loss Function:
The loss of the Generator includes the adversarial loss (GAN Loss) and the L1 loss (L1 Loss, multiplied by a weight of 50)
The loss of the Discriminator (GAN Loss)

### Batch Size: 16

### Number of Training Epochs: 200

### Weight Decay: 1e - 5

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

- [My model](pix2pix_model_epoch_800.pth) 

## Results

the predicted image is as follows:


### [The result predicted by the training set]
<img src="results/train_results.png" alt="train_results" width="800">

### [The result predicted by the validation set]
<img src="results/val_results.png" alt="val_results" width="800">
