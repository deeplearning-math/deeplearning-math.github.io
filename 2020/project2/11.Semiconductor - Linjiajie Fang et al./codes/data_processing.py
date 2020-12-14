import os
import numpy as np
import pandas as pd
import random
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from scipy import misc

PATH = '/Users/fanglinjiajie/locals/datasets/semiconductor/data/'
DEFECT_AREA = pd.read_csv(PATH + 'defect_area.csv')

defect_path = PATH + 'train/defect/'
good_path = PATH + 'train/good/'
test_path = PATH + 'test/'

defect_list = os.listdir(defect_path)
good_list = os.listdir(good_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sample_size = 1000


def get_defect_area(file):
    """

    :param file: e.g. 'SOT23DUMMY01_04-APG_ITIS_H52_1_111_4.bmp'
    :return:
    """
    f = file[:-4]
    if len(DEFECT_AREA[DEFECT_AREA['id'] == f]) > 0:
        return DEFECT_AREA[DEFECT_AREA['id'] == f]['x_1'].item(), \
               DEFECT_AREA[DEFECT_AREA['id'] == f]['y_1'].item(), \
               DEFECT_AREA[DEFECT_AREA['id'] == f]['x_2'].item(), \
               DEFECT_AREA[DEFECT_AREA['id'] == f]['y_2'].item()
    return


def imgshow(array, defect_area=None):
    """
    :param array:
    :param defect_area: e.g. = (87, 137, 118, 174)
    :return:
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(np.transpose(array, (1, 2, 0)), cmap='gray')
    if defect_area:
        x1, y1, x2, y2 = defect_area
        anchor = (x1, y1)
        h = y2 - y1
        w = x2 - x1
        ax.add_patch(plt.Rectangle(anchor, w, h, edgecolor='red', facecolor='none'))
        ax.set_title('defect')

    fig.show()


def imgs2torch(imgs: list, path: str):
    """
    :param path:
    :param imgs: list of file name
    :return:
    """
    tensor = np.zeros((len(imgs), 1, 267, 275))
    for i, f in enumerate(imgs):
        tensor[i] = np.array(Image.open(path + f).resize(size=(275, 267)))[np.newaxis, :, :]
    return torch.FloatTensor(tensor) / 255


def boostrap_training_data(sample_size, shuffle=True):
    defect_sample = imgs2torch(random.choices(defect_list, k=sample_size), path=defect_path)
    good_sample = imgs2torch(random.choices(good_list, k=sample_size), path=good_path)
    train_x = torch.cat([defect_sample, good_sample], dim=0)
    train_y = torch.cat([torch.ones(sample_size), torch.zeros(sample_size)])
    indices = list(range(sample_size * 2))
    if shuffle:
        random.shuffle(indices)
    return (train_x[indices]).to(device), train_y[indices].to(device)


def compute_iou(box1, box2):
    '''Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
    Args:
      box1: (tensor) bounding boxes, sized [N,4].
      box2: (tensor) bounding boxes, sized [N,4].
      box1 = torch.tensor([[0.,0.,10.,10.], [0.,0.,10.,10.], [0.,0.,10.,10.], [0.,0.,10.,10.]])
     box2 = torch.tensor([[0.,0.,10.,10.], [5.,5.,6.,6.], [5.,5.,15.,15.], [5.,-5.,15.,5.]])
    Return:
      (tensor) iou, sized [N,1].
    '''
    lt = torch.max(box1[:, :2], box2[:, :2])

    rb = torch.min(box1[:, 2:], box2[:, 2:])

    wh = rb - lt
    wh[wh < 0] = 0  # clip at 0
    inter = wh[:, 0] * wh[:, 1]

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    iou = inter / (area1 + area2 - inter)
    return iou


def generate_defect_yolo_data():
    data = []
    for i in defect_list:
        area = get_defect_area(i)
        if area:
            x1, y1, x2, y2 = area
            x1, y1, x2, y2 = x1/267, y1/275, x2/267, y2/275
            x_grid, y_grid = int(8 * (x1 + x2)/2), int(8 * (y1 + y2)/2)


if __name__ == '__main__':
    im = np.array(Image.open(PATH + 'train/defect/WEL925224H1A_01-APG_ITIS_H09_2_15_2.bmp'))
    plt.imshow(im, cmap='gray')
    plt.show()

    file = random.choice(defect_list)
    img = np.array(Image.open(PATH + 'train/defect/' + file))[np.newaxis, :, :]
    defect_area = get_defect_area(file)
    imgshow(img, defect_area)

    """
    Create training data: 1: defect, 0: good
    """

    train_x = np.zeros((len(defect_list) + len(good_list), 1, 267, 275))

    for i, f in tqdm(enumerate(defect_list)):
        train_x[i] = np.array(Image.open(PATH + 'train/defect/' + f).resize(size=(275, 267)))[np.newaxis, :, :]

    for i, f in tqdm(enumerate(good_list)):
        train_x[i + len(defect_list)] = np.array(Image.open(PATH + 'train/good/' + f))[np.newaxis, :, :]

    train_y = np.concatenate([np.ones(len(defect_list)), np.zeros(len(good_list))])

    indices = list(range(len(train_y)))
    random.shuffle(indices)

    img_names = defect_list + good_list
    img_names = [img_names[i] for i in indices]

    TRAIN_X = train_x[indices] / 255
    TRAIN_Y = train_y[indices]

    """
    testing data
    """
    test_list = os.listdir(PATH + 'test')
    test_x = np.zeros((len(test_list) + len(good_list), 1, 267, 275))
    for i, f in tqdm(enumerate(test_list)):
        test_x[i] = np.array(Image.open(PATH + 'test/' + f).resize(size=(275, 267)))[np.newaxis, :, :]
