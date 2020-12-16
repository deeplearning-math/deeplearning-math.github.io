import matplotlib.pyplot as plt
import numpy as np
from cv2 import cv2
import os


# black edge removal
crop_image = lambda img, x0, y0, w, h: img[y0:y0+h, x0:x0+w]  # Define the cutting function for subsequent cutting

def rotate_image(img, angle, h,w, crop):
    """
    angle: rotation angle
    crop: doing crop or not,bool
    """
    #w, h = img.shape[:2]
    # period of the rotation angle is 360 
    angle = angle % 360
    # affine transformation matrix
    M_rotation = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    # image after rotation
    img_rotated = cv2.warpAffine(img, M_rotation, (w, h))

    # remove the black edge
    if crop:
        # period of cropping angle is 180
        angle_crop = angle % 180
        if angle > 90:
            angle_crop = 180 - angle_crop
        # convert angle to radian
        theta = angle_crop * np.pi / 180
        # calculate aspect ratio
        hw_ratio = float(h) / float(w)
        # numerator term of cutting edge length coefficient
        tan_theta = np.tan(theta)
        numerator = np.cos(theta) + np.sin(theta) * np.tan(theta)

        # calculate the denominator terms related to aspect ratio
        r = hw_ratio if h > w else 1 / hw_ratio
        # denominator
        denominator = r * tan_theta + 1
        # side length factor
        crop_mult = numerator / denominator

        # get the cropping region
        w_crop = int(crop_mult * w)
        h_crop = int(crop_mult * h)
        x0 = int((w - w_crop) / 2)
        y0 = int((h - h_crop) / 2)
        img_rotated = crop_image(img_rotated, x0, y0, w_crop, h_crop)
    return img_rotated


# get images in file and images augument
def get_files(file_dir):
    if not file_dir[-1] in '/\\':
        file_dir = file_dir + '\\'
    image_list = []
    imgs = []
    image_list = [file_dir + file_name for file_name in os.listdir(file_dir)]
    
    for file_name in image_list:
        extension = file_name.split(".")[-1]
        if not extension in ['jpg', 'jpeg', 'bmp']:
            continue

        img = cv2.imread(file_name, 0)
        img = cv2.resize(img, (224,224))
        # resize the shape
        h, w = img.shape
        

        # give new name for augument images
        file_name_listed = file_name.split(".")
        if(file_name_listed[-2].find("_new") == -1):
            imgs.append(img)
            # horizontal flip
            h_flip = cv2.flip(img,1)
            file_name_listed[-2] = file_name_listed[-2] + "_new_1"
            new_file_name = '.'.join(file_name_listed)
            cv2.imwrite(new_file_name, h_flip)
            imgs.append(h_flip)
            # Remove the black edge and rotate 15 degrees
            image_rotated = rotate_image(img, 15, h, w , True)
            file_name_listed[-2] = file_name_listed[-2] + "_new_2"
            new_file_name = '.'.join(file_name_listed)
            cv2.imwrite(new_file_name, image_rotated)
            imgs.append(image_rotated)
            # Remove the black edge and rotate -15 degrees
            image_rotated2 = rotate_image(img, 345, h, w , True)
            file_name_listed[-2] = file_name_listed[-2] + "_new_3"
            new_file_name = '.'.join(file_name_listed)
            cv2.imwrite(new_file_name, image_rotated2)
            imgs.append(image_rotated2)
    return np.array(imgs)


if __name__ == '__main__':
    train_dir = "/semiconductor2/val/defect"
    imgs = get_files(train_dir)
    
    # resize the shape
    # h, w, _ = img.shape
    # horizontal flip
    # h_flip = cv2.flip(img,1)
    # vertical flip
    # v_flip = cv2.flip(img,0)
    # horizontal vertical flip
    # hv_flip = cv2.flip(img,-1)

    # 15 degree rotation
    # rows, cols, _ = img.shape
    # M = cv2.getRotationMatrix2D((cols/2, rows/2), 15, 1)
    # rotation_15 = cv2.warpAffine(img, M, (cols, rows))

    # -15 degree rotation
    # rows, cols, _ = img.shape
    # M2 = cv2.getRotationMatrix2D((cols/2, rows/2), 345, 1)
    # rotation_135 = cv2.warpAffine(img, M2, (cols, rows))

    # 45 degree rotation
    # M = cv2.getRotationMatrix2D((cols/2, rows/2), 45, 1)
    # rotation_45 = cv2.warpAffine(img, M,(cols, rows))

    # Remove the black edge and rotate 15 degrees
    #image_rotated = rotate_image(img, 15, True)
    # Remove the black edge and rotate -15 degrees
    #image_rotated2 = rotate_image(img, 345, True)

    # # transform to right down
    # mat_shift = np.float32([[1,0,10], [0,1,10]])
    # img_RD = cv2.warpAffine(img, mat_shift, (h, w))
    # # transform to left up
    # mat_shift2 = np.float32([[1, 0, -10], [0, 1, -10]])
    # img_LU = cv2.warpAffine(img, mat_shift2, (h, w))