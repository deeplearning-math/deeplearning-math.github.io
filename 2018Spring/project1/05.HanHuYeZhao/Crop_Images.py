import numpy as np
from PIL import Image, ImageStat
from os import listdir
import os
import shutil
from torchvision import transforms

## HU WEI @23Feb.2018
class crop_Images(object):

    def __init__(self, pictures_dir, labels, low_var_filter=False, low_var_threshold=0):
        self.source_dir = pictures_dir
        self.destination_dir = 'data/train'
        self.labels = labels
        self.filelist = sorted(listdir(self.source_dir), key=(lambda x:int(x.split('.')[0])))
        self.low_var_filter = low_var_filter
        self.low_var_threshold = low_var_threshold
        self.num_ignore = 0
        self.num_pictures = 0

    def random_crop(self, crop_size = (256,256), n_multiple = 2):
        rand_crop = transforms.Compose([
            transforms.RandomCrop(crop_size),
        ])
        for i in range(len(self.filelist)):
            img = Image.open(self.source_dir + '/' + self.filelist[i])
            n_pixels = img.size[0]*img.size[1]
            n_crop = n_multiple * int(n_pixels / crop_size[0] / crop_size[1])
            file_path = self.destination_dir + '/' + self.labels[self.filelist[i].split('.')[0]]
            img_count = 0
            try:
                if not os.path.exists(file_path):
                    print('folder: ', file_path, 'does not exists, creating')
                    os.makedirs(file_path)
            except IOError as e:
                print('Error: ', e)
            except Exception as e:
                print('Error:', e)
            while img_count < n_crop:
                crop_img = rand_crop(img)
                if self.low_var_filter and sum(ImageStat.Stat(crop_img).var) < self.low_var_threshold:
                    continue
                crop_img.save(file_path + '/' + self.filelist[i].split('.')[0] + '-' + str(img_count) + '.tif')
                img_count += 1
            self.num_pictures += n_crop
        self.__move_disputed__()
        print('totally %d pictures created ' % self.num_pictures)
        self.num_pictures = 0

    def sequential_crop(self, crop_size=(256, 256), offset=(180, 180)):
        w_crop, h_crop = crop_size
        w_offset, h_offset = offset
        for i in range(len(self.filelist)):
            img = Image.open(self.source_dir + '/' + self.filelist[i])
            w, h = img.size
            file_path = os.path.join(self.destination_dir, self.labels[self.filelist[i].split('.')[0]])
            try:
                if not os.path.exists(file_path):
                    print('Folder:', file_path, 'does not exist, creating')
                    os.makedirs(file_path)
            except IOError as e:
                print('Error: ', e)
            except Exception as e:
                print('Error:', e)
            w_p, h_p = (0, 0)
            while w_p + w_crop < w:
                while h_p + h_crop < h:
                    img_crop = img.crop((w_p, h_p, w_p + w_crop, h_p + h_crop))
                    if self.low_var_filter and sum(ImageStat.Stat(img_crop).var) < self.low_var_threshold:
                        self.num_ignore += 1
                        h_p = h_p + h_offset
                        continue
                    img_crop.save(
                        file_path + '/' + self.filelist[i].split('.')[0] + '-' + str(w_p) + '-' + str(h_p) + '.tif')
                    self.num_pictures += 1
                    h_p = h_p + h_offset
                h_p = 0
                w_p = w_p + w_offset
        self.__move_disputed__()
        print('totally %d pictures created, we ignore %d low variance pictures' % (self.num_pictures, self.num_ignore))
        self.num_pictures = 0
        self.num_ignore = 0

    def shuffle_validation(self):
        data_dir = self.destination_dir.split('/')[0]
        train_dir = os.path.join(data_dir, 'train')
        val_dir = os.path.join(data_dir, 'val')
        train_val_dir = {1: train_dir, 2: val_dir}
        for i in train_val_dir:
            try:
                if not os.path.exists(train_val_dir[i]):
                    print('Folder:', train_val_dir[i], 'does not exist, creating')
                    os.makedirs(train_val_dir[i] + '/' + 'authentic')
                    os.makedirs(train_val_dir[i] + '/' + 'fake')
            except IOError as e:
                print('Fail: ', e)
            except Exception as e:
                print('Fail: ', e)
        authentic_list = [key for key in self.labels if self.labels[key] == 'authentic']
        fake_list = [key for key in self.labels if self.labels[key] == 'fake']
        val_authentic = np.random.choice(authentic_list)
        val_fake = np.random.choice(fake_list)
        train_authentic_dir = os.path.join(train_dir, 'authentic')
        train_fake_dir = os.path.join(train_dir, 'fake')
        val_authentic_dir = os.path.join(val_dir, 'authentic')
        val_fake_dir = os.path.join(val_dir, 'fake')
        # move files from train folder to validation
        for files in listdir(train_authentic_dir):
            if files.startswith(str(val_authentic) + '-'):
                shutil.move(os.path.join(train_authentic_dir, files), val_authentic_dir)
        for files in listdir(train_fake_dir):
            if files.startswith(str(val_fake) + '-'):
                shutil.move(os.path.join(train_fake_dir, files), val_fake_dir)
        # move files from validation folder to train
        for files in listdir(val_authentic_dir):
            if not files.startswith(str(val_authentic) + '-'):
                shutil.move(os.path.join(val_authentic_dir, files), train_authentic_dir)
        for files in listdir(val_fake_dir):
            if not files.startswith(str(val_fake) + '-'):
                shutil.move(os.path.join(val_fake_dir, files), train_fake_dir)
        print('take %s and %s as validation pictures' % (val_authentic, val_fake))

    def __move_disputed__(self):
        if os.path.exists(os.path.join(self.destination_dir, 'disputed')):
            if os.path.exists(os.path.join(self.destination_dir, '..', 'disputed')):
                shutil.rmtree(os.path.join(self.destination_dir, 'disputed'))
            else:
                shutil.move(os.path.join(self.destination_dir, 'disputed'), os.path.join(self.destination_dir, '..'))





