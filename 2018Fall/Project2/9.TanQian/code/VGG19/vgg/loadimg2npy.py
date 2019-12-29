"""
Simple tester for the vgg19_trainable
"""
# import utils
# import os, os.path

# path = "/vgg19/train"

# for f in os.listdir(path):
#     img1 = utils.load_image("./test_data/tiger.jpeg")

import glob
import utils
import numpy as np

img_arr = []
i=0

for filename in glob.glob('all_tests/*.jpg'):
    img = utils.load_image(filename)
    img_arr.append(img)
    i = i+1
    print(i)

np.save('test.npy',img_arr)