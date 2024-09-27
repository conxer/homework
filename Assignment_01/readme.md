# 作业 1 - 图像变形

### 在本次作业中，您将看到对图像的基本变换和基于点的变形。

### 本文用到的资源:
- [Teaching Slides](https://rec.ustc.edu.cn/share/afbf05a0-710c-11ef-80c6-518b4c8c0b96) 
- [Paper: Image Deformation Using Moving Least Squares](https://people.engr.tamu.edu/schaefer/research/mls.pdf)
- [Paper: Image Warping by Radial Basis Functions](https://www.sci.utah.edu/~gerig/CS6640-F2010/Project3/Arad-1995.pdf)
- [OpenCV Geometric Transformations](https://docs.opencv.org/4.x/da/d6e/tutorial_py_geometric_transformations.html)
- [Gradio: 一个好用的网页端交互GUI](https://www.gradio.app/)

### 1. 基本图像几何变换（缩放/旋转/平移）。
本文的几何变换是基于仿射变换实现的。

### 2. 基于点的图像变形。
本文的图像变形是基于径向基函数（RBF）实现的，径向基函数（Radial Basis Function，简称RBF）是一种用于多变量插值的方法，广泛应用于机器学习和信号处理领域。具体算法见[Paper: Image Warping by Radial Basis Functions](https://www.sci.utah.edu/~gerig/CS6640-F2010/Project3/Arad-1995.pdf)
本文所用的是平方倒数径向基函数。
#### 平方倒数径向基函数（Inverse Multiquadric Radial Basis Function）
平方倒数径向基函数（Inverse Multiquadric Radial Basis Function）是一种径向基函数，其数学表达式通常如下所示：
    φ(r) = 1 / √(d + r²)
其中：   φ(r):径向基函数的值。
        r: 从原点（或者某个中心点）到输入点的欧氏距离。
        d:</是一个形状参数，通常是一个正常数。
    这个函数的特点是，当距离 r 增加时，函数值会逐渐减小，并且随着距离的增加，函数值的减小速度会逐渐变慢。平方倒数径向基函数的一个关键特性是它能够处理多维输入空间中的点，并且它的函数值仅依赖于输入点之间的距离。这使得它在处理空间数据或者需要捕捉输入特征之间复杂关系的问题时非常有用。在实际应用中，参数 c 的选择对函数的行为有很大影响。不同的 c 值会导致函数的平滑度和局部敏感性有所不同，因此在设计模型时需要仔细选择这个参数。



## 要求

要安装要求：

```安装
python -m pip install -r requirements.txt
```

## 运行

要运行基本转换，请运行：

```basic
python run_global_transform.py
```

要运行点引导变换，请运行：

```point
python run_point_transform.py
```

## 结果
### 基本变换

<img src="pics/global-transform .gif" alt="alt text" width="800">

### 点引导变形：
<img src="pics/teaser.png" alt="alt text" width="800">
<img src="pics/point-transform.gif" alt="alt text" width="800">

##  致谢

>📋 感谢[Paper: Image Warping by Radial Basis Functions](https://www.sci.utah.edu/~gerig/CS6640-F2010/Project3/Arad-1995.pdf)提出的算法.
