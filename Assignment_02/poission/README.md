# Implement Poisson Image Editing with PyTorch

This repository is the official implementation of [Implement Poisson Image Editing with PyTorch](https://www.cs.jhu.edu/~misha/Fall07/Papers/Perez03.pdf)


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Operation

To operation the model(s) in the paper, run this command:

```
python run_blending_gradio.py 
```
Then follow the video operation

## Results

Our model achieves the following image fusion:

### 1、[Blend the equation into the sky background]
![source](Assignment_02/poission/data_poission/equation/source.png)
![target](Assignment_02/poission/data_poission/equation/target.jpg)
![results](Assignment_02/poission/data_poission/equation/results.png)
### 2、[Mona Lisa changes face]
![source](Assignment_02/poission/data_poission/monolisa/source.png)
![target](Assignment_02/poission/data_poission/monolisa/target.png)
![results](Assignment_02/poission/data_poission/monolisa/results.png)
### 3、[Sharks pose for a photo]
![source](Assignment_02/poission/data_poission/monolisa/water.jpg)
![target](Assignment_02/poission/data_poission/monolisa/water.jpg)
![results](Assignment_02/poission/data_poission/monolisa/water.png)
