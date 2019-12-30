# Target Architecture - RAM for augmented Dataset
An attempt to implement the idea from Recurrent Models of Visual Attention, Volodymyr Mnih, Nicolas Heess, Alex Graves, Koray Kavukcuoglu, https://arxiv.org/abs/1406.6247

The idea is to use hybrid loss to approximate policy gradient result, detailed explained in [this blog](https://medium.com/@tianyu.tristan/visual-attention-model-in-deep-learning-708813c2912c)

Code extended from [this repo](https://github.com/zhongwen/RAM), with glimpse sensor changed to support multiple glimpse scales
