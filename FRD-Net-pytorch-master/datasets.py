import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class DriveDataset(Dataset):
    def __init__(self, root: str, train: bool, transforms=None):
        super(DriveDataset, self).__init__()
        data_root = os.path.join(root, "aug" if train else "test")
        assert os.path.exists(data_root), f"path '{data_root}' does not exists."
        self.transforms = transforms
        img_names = [i for i in os.listdir(os.path.join(data_root, "images")) if i.endswith(".tif")]
        self.img_list = [os.path.join(data_root, "images", i) for i in img_names]
        self.manual = [os.path.join(data_root, "1st_manual", i.split("_")[0] + "_manual1.gif")
                       for i in img_names]
        # check files
        for i in self.manual:
            if os.path.exists(i) is False:
                print(f"file {i} does not exists.")

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx]).convert('RGB')
        mask = Image.open(self.manual[idx]).convert('L')
        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        return img, mask

    def __len__(self):
        return len(self.img_list)


    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs


class Chasedb1Datasets:
    def __init__(self, root: str, train: bool, transforms=None):
        super().__init__()
        if train:
            data_root = os.path.join(root, "aug" if train else "test")
            assert os.path.exists(data_root), f"path '{data_root}' does not exists."
            self.transforms = transforms
            img_names = [i for i in os.listdir(os.path.join(data_root, "images")) if i.endswith(".jpg")]
            self.img_list = [os.path.join(data_root, "images", i) for i in img_names]
            self.manual = [os.path.join(data_root, "1st_label", i.split(".")[0] + "_1stHO.png")
                           for i in img_names]
        # check files
        for i in self.manual:
            if os.path.exists(i) is False:
                print(f"file {i} does not exists.")

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx]).convert('RGB')
        manual = Image.open(self.manual[idx]).convert('L')
        if self.transforms is not None:
            img, manual = self.transforms(img, manual)

        return img, manual

    def __len__(self):
        return len(self.img_list)


#      下面两个函数是图片弄成batch批次、 FCN里面讲了
#     @staticmethod
#     def collate_fn(batch):
#         images, targets, thicks, thins = list(zip(*batch))
#         batched_imgs = cat_list(images, fill_value=0)
#         batched_targets = cat_list(targets, fill_value=255)
#         return batched_imgs, batched_targets
#
#
# def cat_list(images, fill_value=0):
#     max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
#     batch_shape = (len(images),) + max_size
#     batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
#     for img, pad_img in zip(images, batched_imgs):
#         pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
#     return batched_imgs


class STAREDataset(Dataset):
    def __init__(self, root: str, train: bool, transforms=None):
        super(STAREDataset, self).__init__()
        data_root = os.path.join(root, "aug" if train else "test")
        assert os.path.exists(data_root), f"path '{data_root}' does not exists."
        self.transforms = transforms
        img_names = [i for i in os.listdir(os.path.join(data_root, "images1")) if i.endswith(".ppm")]
        self.img_list = [os.path.join(data_root, "images1", i) for i in img_names]
        self.manual = [os.path.join(data_root, "1st_labels_ah", i.split(".")[0] + ".ah.ppm")
                       for i in img_names]
        # check files
        for i in self.manual:
            if os.path.exists(i) is False:
                print(f"file {i} does not exists.")

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx]).convert('RGB')
        # 值为0和255
        mask = Image.open(self.manual[idx]).convert('L')
        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        return img, mask

    def __len__(self):
        return len(self.img_list)

# 下面两个函数是图片弄成batch批次、 FCN里面讲了
#     @staticmethod
#     def collate_fn(batch):
#         images, targets = list(zip(*batch))
#         batched_imgs = cat_list(images, fill_value=0)
#         batched_targets = cat_list(targets, fill_value=255)
#         return batched_imgs, batched_targets
#
#
# def cat_list(images, fill_value=0):
#     max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
#     batch_shape = (len(images),) + max_size
#     batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
#     for img, pad_img in zip(images, batched_imgs):
#         pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
#     return batched_imgs


class HRFDataset(Dataset):
    def __init__(self, root: str, train: bool, transforms=None):
        super(HRFDataset, self).__init__()
        data_root = os.path.join(root, "aug" if train else "test")
        assert os.path.exists(data_root), f"path '{data_root}' does not exists."
        self.transforms = transforms
        img_names = [i for i in os.listdir(os.path.join(data_root, "images")) if i.endswith(".tif")]
        self.img_list = [os.path.join(data_root, "images", i) for i in img_names]
        self.manual = [os.path.join(data_root, "manual1", i.split(".")[0] + ".tif")
                       for i in img_names]
        # check files
        for i in self.manual:
            if os.path.exists(i) is False:
                print(f"file {i} does not exists.")

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx]).convert('RGB')
        # 值为0和255
        mask = Image.open(self.manual[idx]).convert('L')
        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        return img, mask

    def __len__(self):
        return len(self.img_list)

#     # 下面两个函数是图片弄成batch批次、 FCN里面讲了
#     @staticmethod
#     def collate_fn(batch):
#         images, targets = list(zip(*batch))
#         batched_imgs = cat_list(images, fill_value=0)
#         batched_targets = cat_list(targets, fill_value=255)
#         return batched_imgs, batched_targets
#
#
# def cat_list(images, fill_value=0):
#     max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
#     batch_shape = (len(images),) + max_size
#     batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
#     for img, pad_img in zip(images, batched_imgs):
#         pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
#     return batched_imgs
