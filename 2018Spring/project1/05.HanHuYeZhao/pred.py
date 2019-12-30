from os import listdir
from PIL import Image
from collections import defaultdict
from torch.autograd import Variable
from torchvision import transforms
import torch
import os
use_gpu = torch.cuda.is_available()

## HU WEI @23Feb.2018
class predictor(object):

    def __init__(self, model):
        self.model = model
        self.results = defaultdict(list)
        self.images = defaultdict(list)
        self.folder = None
        self.prob = {}

    def __load_images__(self):
        self.images = defaultdict(list)
        # We shall load data to self.images so that it is a dictionary of Variables of Tensors
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        for files in listdir(self.folder):
            img = trans(Image.open(os.path.join(self.folder, files)))[0:3, :, :].unsqueeze(0)
            key = files.split('-')[0]
            self.images[key].append(img)
         #Now merge Tensors
        # for key in self.images:
        #     self.images[key] = torch.cat(self.images[key], 0)
        # print('loading success')

    # TO DO: Predict Vast Labels parallel to Speed up, Currently, out of memory
    def prediction(self, folder):
        self.folder = folder
        self.results = defaultdict(list)
        self.prob = {}
        self.__load_images__()
        for key in self.images:
            # self.results[key] = self.model(Variable(self.images[key].cuda()))
            for img in self.images[key]:
                if use_gpu:
                    img = img.cuda()
                _, pred = torch.max(self.model(Variable(img)).data, 1)
                self.results[key].append(pred)
            self.results[key] = torch.cat(self.results[key])
            self.prob[key] = 1 - torch.mean(self.results[key].float())

