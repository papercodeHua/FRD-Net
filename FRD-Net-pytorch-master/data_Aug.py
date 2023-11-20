import logging
import os
import random
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFile
import transforms as T

logger = logging.getLogger(__name__)
ImageFile.LOAD_TRUNCATED_IMAGES = True


class DataAugmentation:

    def __init__(self):
        pass

    @staticmethod
    def openImage(image):
        return Image.open(image)

    @staticmethod
    def randomRotation(image, label, mask, mode=Image.BICUBIC):
        random_angle = np.random.randint(1, 360)
        return image.rotate(random_angle, mode), label.rotate(random_angle, Image.NEAREST), mask.rotate(random_angle, Image.NEAREST)

    @staticmethod
    def randomCrop(image, label, mask):
        image_width = image.size[0]
        image_height = image.size[1]
        crop_win_size = np.random.randint(40, 68)
        random_region = (
            (image_width - crop_win_size) >> 1, (image_height - crop_win_size) >> 1, (image_width + crop_win_size) >> 1,
            (image_height + crop_win_size) >> 1)
        return image.crop(random_region), label, mask

    @staticmethod
    def randomColor(image, label, mask):
        random_factor = np.random.randint(15, 21) / 10.
        color_image = ImageEnhance.Color(image).enhance(random_factor)
        random_factor = np.random.randint(15, 21) / 10.
        brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)
        random_factor = np.random.randint(15, 21) / 10.
        contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)
        random_factor = np.random.randint(15, 31) / 10.
        return ImageEnhance.Sharpness(contrast_image).enhance(random_factor), label, mask

    @staticmethod
    def randomGaussian(image, label, mask, mean=0.2, sigma=0.3):
        def gaussianNoisy(im, mean=0.2, sigma=0.3):
            for _i in range(len(im)):
                im[_i] += random.gauss(mean, sigma)
            return im

        img = np.array(image)
        img.flags.writeable = 1
        width, height = img.shape[:2]
        img_r = gaussianNoisy(img[:, :, 0].flatten(), mean, sigma)
        img_g = gaussianNoisy(img[:, :, 1].flatten(), mean, sigma)
        img_b = gaussianNoisy(img[:, :, 2].flatten(), mean, sigma)
        img[:, :, 0] = img_r.reshape([width, height])
        img[:, :, 1] = img_g.reshape([width, height])
        img[:, :, 2] = img_b.reshape([width, height])
        return Image.fromarray(np.uint8(img)), label, mask

    @staticmethod
    def saveImage(image, path):
        image.save(path)


def ntsc_grayscale(image):
    ntsc_weights = np.array([0.299, 0.587, 0.114])
    grayscale_image = np.dot(image[..., :3], ntsc_weights)
    grayscale_image = grayscale_image.astype(np.uint8)
    return grayscale_image


def clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    equalized_image = clahe.apply(image)
    return equalized_image


def gamma_correction(image, gamma=1.0):
    gamma_corrected = np.power(image / 255.0, gamma) * 255.0
    gamma_corrected = gamma_corrected.astype(np.uint8)
    return gamma_corrected



def normalize(image):
    mean = np.mean(image)
    var = np.mean(np.square(image - mean))

    image = (image - mean) / np.sqrt(var)

    return image


def dataset_pro(imgs, label, mask):
    imgs = imgs
    label = label
    mask = mask
    return imgs, label, mask


def imageOps(func_name, image, label, mask, img_des_path, label_des_path, mask_des_path, img_file_name, label_file_name,
             mask_file_name, times=1):
    funcMap = {"randomRotation": DataAugmentation.randomRotation,
               "randomRotation1": dataset_pro,
               "randomCrop": DataAugmentation.randomCrop,
               "randomColor": DataAugmentation.randomColor,
               "randomGaussian": DataAugmentation.randomGaussian
               }
    if funcMap.get(func_name) is None:
        logger.error("%s is not exist", func_name)
        return -1

    for i in range(0, times):
        new_image, new_label, new_mask = funcMap[func_name](image, label, mask)

        new_image = cv2.cvtColor(np.asarray(new_image), cv2.COLOR_RGB2BGR)
        new_image = ntsc_grayscale(new_image)
        new_image = clahe(new_image)
        new_image = gamma_correction(new_image, 1.4)
        new_image = Image.fromarray(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))

        DataAugmentation.saveImage(new_image, os.path.join(img_des_path, str(i) + img_file_name))
        DataAugmentation.saveImage(new_label, os.path.join(label_des_path, str(i) + label_file_name))
        DataAugmentation.saveImage(new_mask, os.path.join(mask_des_path, str(i) + mask_file_name))


opsList = {"randomRotation1"}


def threadOPS(img_path, new_img_path, label_path, new_label_path, mask_path, new_mask_path):
    img_names = os.listdir(img_path)
    label_names = os.listdir(label_path)
    mask_names = os.listdir(mask_path)


    img_num = len(img_names)
    label_num = len(label_names)
    mask_num = len(mask_names)

    assert img_num == label_num or img_num == mask_num or mask_num == label_num, f"图片和标签和掩码数量不一致"
    num = img_num

    for i in range(num):
        img_name = img_names[i]
        label_name = label_names[i]
        mask_name = mask_names[i]

        tmp_img_name = os.path.join(img_path, img_name)
        tmp_label_name = os.path.join(label_path, label_name)
        tmp_mask_name = os.path.join(mask_path, mask_name)


        image = DataAugmentation.openImage(tmp_img_name)

        label = DataAugmentation.openImage(tmp_label_name)

        mask = DataAugmentation.openImage(tmp_mask_name)

        for ops_name in opsList:
            imageOps(ops_name, image, label, mask, new_img_path, new_label_path, new_mask_path, img_name,
                     label_name, mask_name,)


# Please modify the path
if __name__ == '__main__':
    # DRIVE
    file_path1 = "DRIVE/aug/images"
    file_path2 = "DRIVE/aug/1st_manual"
    file_path3 = "DRIVE/aug/mask"
    if not os.path.exists(file_path1 or file_path2 or file_path3):
        os.makedirs(file_path1)
        os.makedirs(file_path2)
        os.makedirs(file_path3)
    threadOPS("DRIVE/train/images",  # set your path of training images
              "DRIVE/aug/images",
              "DRIVE/train/1st_manual",  # set your path of training labels
              "DRIVE/aug/1st_manual",
              "DRIVE/train/mask",  # set your path of training mask
              "DRIVE/aug/mask")

