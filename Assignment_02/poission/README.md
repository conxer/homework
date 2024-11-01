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

## Results

Our model achieves the following image fusion:

### 1、[Blend the equation into the sky background]
#### Source Image
<img src="data_poission/equation/source.png" alt="source image" width="800">

#### Target Image
<img src="data_poission/equation/target.jpg" alt="target image" width="800">

#### Results Image
<img src="data_poission/equation/results.png" alt="results image" width="800">

### 2、[Mona Lisa changes face]
#### Source Image
<img src="data_poission/monolisa/source.png" alt="source image" width="800">

#### Target Image
<img src="data_poission/monolisa/target.png" alt="target image" width="800">

#### Results Image
<img src="data_poission/monolisa/results.png" alt="results image" width="800">

### 3、[Sharks pose for a photo]
#### Source Image
<img src="data_poission/water/source.jpg" alt="source image" width="800">

#### Target Image
<img src="data_poission/water/target.jpg" alt="target image" width="800">

#### Results Image
<img src="data_poission/water/results.png" alt="results image" width="800">

