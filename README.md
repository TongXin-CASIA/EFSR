# Serial section electron microscope image registration with enhanced feature learning and structural regression
This repository contains the official implementation of the paper
"Serial section electron microscope image registration with enhanced feature learning and structural regression"

![Python 3.9](https://img.shields.io/badge/python-3.9-green.svg?style=plastic) ![PyTorch 1.10.0](https://img.shields.io/badge/pytorch-1.10.0-green.svg?style=plastic) 
![license MIT](https://img.shields.io/github/license/TongXin-CASIA/EFSR?style=plastic)
## Using the Code
### Requirements
This code has been developed under Python 3.9, PyTorch 1.10, and Ubuntu 16.04.

In addition to the above libraries, the python environment can be set as follows:

```shell
conda create -n FESR
conda activate FESR
pip3 install opencv-python torch
pip3 install scipy pillow scikit-image matplotlib EasyDict
```


### Register two sections
```Register
python pairwise.py --reference test1.png --moving test2.png --output output.png
```

### Register short serial sections
```Register
python serial.py --input_dir input_dir --output_dir output
```

### Datasets in the paper

[CSTCloud](https://pan.cstcloud.cn/s/Ys31sNa6ROg)

### Citation
```
coming soon
````