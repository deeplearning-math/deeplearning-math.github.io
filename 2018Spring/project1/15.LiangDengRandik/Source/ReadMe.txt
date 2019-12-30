[*License]This project is revised from https://github.com/kratzert/finetune_alexnet_with_tensorflow

1. Requirements
Before implete this code, make sure you have download the pretrained weights `bvlc_alexnet.npy`, which you can find [here](http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/) and put the weights in the current working directory.

2. Packages:
- Python 3
- TensorFlow >= 1.12rc0
- Numpy
- tqdm
- sklearn
- pandas
- ggplot
- opencv
- keras
- tensorflow
- pickle

3. Contents
- `alexnet.py`: Class with the graph definition of the AlexNet.
- `datagenerator.py`: Contains a wrapper class for the new input pipeline.
- `caffe_classes.py`: List of the 1000 class names of ImageNet (copied from (http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/)).
- `mnist_feature_Pretrained_Alexnet_ImageNet.ipynb` : ipython notebook on how to extract features using pretrained AlexNet and classify with supervised learning


