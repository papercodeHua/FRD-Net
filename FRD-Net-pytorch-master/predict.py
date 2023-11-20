import time
from torchvision import transforms
import numpy as np
from PIL import Image
import os
from torch import nn
import torch.utils.data
import transforms as T
from datasets import DriveDataset
from model.FRD_Net import FRD_Net
from train_utils.train_and_eval import evaluate
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from train_utils.pad import InputPadder
from train_utils.disturtd_utils import ConfusionMatrix


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


class SegmentationPresetEval:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def main():
    classes = 1  # exclude background
    weights_path = "save_weights/best_model.pth"
    img_path = "DRIVE/test/images/005_test.tif"
    roi_mask_path = "DRIVE/test/mask/05_test_mask.gif"
    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    assert os.path.exists(img_path), f"image {img_path} not found."
    assert os.path.exists(roi_mask_path), f"image {roi_mask_path} not found."

    num_classes = 1
    mean1 = (0.215, 0.215, 0.215)
    std1 = (0.151, 0.151, 0.151)

    # get devices
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    # create model
    model = FRD_Net(in_channels=3, num_classes=classes, base_c=32)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=[1])

    # load weights
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
    # model.load_state_dict(torch.load(weights_path)['model'])
    model.to(device)
    val_dataset = DriveDataset('DRIVE',
                                   train=False,
                                   transforms=SegmentationPresetEval(mean1, std1))
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             num_workers=1,
                                             pin_memory=True,
                                             collate_fn=val_dataset.collate_fn)
    acc, se, sp, F1, mIou, pr, AUC_ROC = evaluate(model, val_loader, device=device, num_classes=num_classes)
    print(f"AUC: {AUC_ROC:.6f}")
    print(f"acc: {acc:.6f}")
    print(f"se: {se:.6f}")
    print(f"sp: {sp:.6f}")
    print(f"mIou: {mIou:.6f}")
    print(f"F1: {F1:.6f}")
    roi_img = Image.open(roi_mask_path).convert('RGB')
    roi_img = np.array(roi_img)

    # load image
    original_img = Image.open(img_path).convert('RGB')

    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=mean1, std=std1)])
    img = data_transform(original_img)
    roi_img = data_transform(roi_img)
    img = torch.unsqueeze(img, dim=0)
    roi_img = torch.unsqueeze(roi_img, dim=0)
    padder = InputPadder(img.shape)
    img, roi_img = padder.pad(img, roi_img)
    model.eval()
    with torch.no_grad():
        # init model
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        t_start = time_synchronized()
        output = model(img.to(device))
        t_end = time_synchronized()
        print("inference time: {}".format(t_end - t_start))

        output[output >= 0.5] = 1
        output[output < 0.5] = 0
        # prediction = output.argmax(1).squeeze(0)
        prediction = output.squeeze(0).squeeze(0)
        prediction = prediction.to("cpu").numpy().astype(np.uint8)
        # prediction = output.to("cpu").numpy().astype(np.uint8)
        prediction[prediction == 1] = 255
        # prediction[roi_img == 0] = 0
        mask = Image.fromarray(prediction)
        mask.save("test_result.png")


def evaluate(model, data_loader, device, num_classes):
    model.eval()
    confmat = ConfusionMatrix(num_classes + 1)
    data_loader = tqdm(data_loader)
    mask = None
    predict = None
    with torch.no_grad():
        for image, target in data_loader:
            image, target = image.to(device), target.to(device) # 传入cuda
            padder = InputPadder(image.shape)
            image, target = padder.pad(image, target)
            image, target = image.to(device), target.to(device)
            # (B,1,H,W)
            output = model(image)
            truth = output.clone()
            output[output >= 0.5] = 1
            output[output < 0.5] = 0

            confmat.update(target.flatten(), output.long().flatten())
            # dice.update(output, target)
            mask = target.flatten() if mask is None else torch.cat((mask, target.flatten()))
            predict = truth.flatten() if predict is None else torch.cat((predict, truth.flatten()))

    mask = mask.cpu().numpy()
    predict = predict.cpu().numpy()
    assert mask.shape == predict.shape, f"维度不对"
    AUC_ROC = roc_auc_score(mask, predict)

    return confmat.compute()[0], confmat.compute()[1], confmat.compute()[2], confmat.compute()[3], confmat.compute()[
        4], confmat.compute()[5], AUC_ROC


if __name__ == '__main__':
    main()
