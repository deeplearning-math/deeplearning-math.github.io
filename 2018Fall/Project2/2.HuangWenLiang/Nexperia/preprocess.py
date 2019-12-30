import numpy as np
import os
from PIL import Image
import pandas as pd

def cropping(image, center_width):
    img = np.array(image)
    if center_width == 224:
        return img
    else:
        margin = (224-center_width)//2
        img_crop = img[margin:(margin+center_width),margin:(margin+center_width)]
        return img_crop

def load_data(center_width=224):
    path = './nexperia/'
    good_0 = [filename for filename in os.walk(path+'train/good_0/')][0][2]
    bad_1  = [filename for filename in os.walk(path+'train/bad_1/')][0][2]
    
    size_good = len(good_0)
    size_bad = len(bad_1)
    size_train = size_good+size_bad
    
    y = np.zeros(size_train)
    y[0:size_bad] = 1
    
    X = np.zeros((size_train, center_width, center_width))
    for i in range(size_bad):
        image = Image.open(path+'train/bad_1/'+bad_1[i])
        X[i,:,:] = cropping(image, center_width)
    for i in range(size_good):
        image = Image.open(path+'train/good_0/'+good_0[i])
        X[i+size_bad,:,:] = cropping(image, center_width)
    
    mask = list(range(0,size_train))
    np.random.shuffle(mask)
    
    val_split = int(0.2*size_train)
    
    X_val = X[mask[:val_split],:,:]
    y_val = y[mask[:val_split]]
    X_train = X[mask[val_split:],:,:]
    y_train = y[mask[val_split:]]
    
    test_id = list(pd.read_csv(path+'sample_submission.csv')['id'])
    size_test = len(test_id)
    X_test = np.zeros((size_test,center_width,center_width))
    for i in range(size_test):
        image = Image.open(path+'test/'+test_id[i]+'.jpg')
        X_test[i,:,:] = cropping(image, center_width)

    return (X_train, y_train), (X_val, y_val), (X_test, test_id)
    
        