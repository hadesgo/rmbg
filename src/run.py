import os
import time
import numpy as np
from skimage import io
import time
from glob import glob
from PIL import Image
import torch, gc
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import torch.optim as optim
import torch.nn.functional as F

from data_loader_cache import normalize, im_reader, im_preprocess
from models import *


class GOSNormalize(object):
    """
    Normalize the Image using torch.transforms
    """

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image = normalize(image, self.mean, self.std)
        return image


transform = transforms.Compose([GOSNormalize([0.5, 0.5, 0.5], [1.0, 1.0, 1.0])])


def load_image(im_path, hypar):
    im = im_reader(im_path)
    im, im_shp = im_preprocess(im, hypar["cache_size"])
    im = torch.divide(im, 255.0)
    shape = torch.from_numpy(np.array(im_shp))
    # make a batch of image, shape
    return transform(im).unsqueeze(0), shape.unsqueeze(0)


def build_model(hypar, device):
    # GOSNETINC(3,1)
    net = hypar["model"]

    # convert to half precision
    if hypar["model_digit"] == "half":
        net.half()
        for layer in net.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.float()

    net.to(device)

    if hypar["restore_model"] != "":
        net.load_state_dict(
            torch.load(
                hypar["model_path"] + "/" + hypar["restore_model"], map_location=device
            )
        )
        net.to(device)
    net.eval()
    return net


def predict(net, inputs_val, shapes_val, hypar, device):
    """
    Given an Image, predict the mask
    """
    net.eval()

    if hypar["model_digit"] == "full":
        inputs_val = inputs_val.type(torch.FloatTensor)
    else:
        inputs_val = inputs_val.type(torch.HalfTensor)

    # wrap inputs in Variable
    inputs_val_v = Variable(inputs_val, requires_grad=False).to(device)

    # list of 6 results
    ds_val = net(inputs_val_v)[0]

    # B x 1 x H x W
    # we want the first one which is the most accurate prediction
    pred_val = ds_val[0][0, :, :, :]

    ## recover the prediction spatial size to the orignal image size
    pred_val = torch.squeeze(
        F.upsample(
            torch.unsqueeze(pred_val, 0),
            (shapes_val[0][0], shapes_val[0][1]),
            mode="bilinear",
        )
    )

    ma = torch.max(pred_val)
    mi = torch.min(pred_val)
    pred_val = (pred_val - mi) / (ma - mi)  # max = 1

    if device == "cuda":
        torch.cuda.empty_cache()

    # it is the mask we need
    return (pred_val.detach().cpu().numpy() * 255).astype(np.uint8)


def inference(image: Image):
    image_path = image

    image_tensor, orig_size = load_image(image_path, hypar)
    mask = predict(net, image_tensor, orig_size, hypar, device)

    pil_mask = Image.fromarray(mask).convert("L")
    im_rgb = Image.open(image).convert("RGB")

    im_rgba = im_rgb.copy()
    im_rgba.putalpha(pil_mask)

    return im_rgba


if __name__ == "__main__":
    # Set Parameters
    # paramters for inferencing
    hypar = {}

    ## load trained weights from this path
    hypar["model_path"] = "./saved_models"

    ## name of the to-be-loaded weights
    hypar["restore_model"] = "isnet-general-use.pth"
    ## indicate if activate intermediate feature supervision
    hypar["interm_sup"] = False

    ##  choose floating point accuracy --
    ## indicates "half" or "full" accuracy of float number
    hypar["model_digit"] = "full"
    hypar["seed"] = 0

    ## cached input spatial resolution, can be configured into different size
    hypar["cache_size"] = [1024, 1024]

    ## data augmentation parameters ---
    ## mdoel input spatial size, usually use the same value hypar["cache_size"], which means we don't further resize the images
    hypar["input_size"] = [1024, 1024]

    ## random crop size from the input, it is usually set as smaller than hypar["cache_size"], e.g., [920,920] for data augmentation
    hypar["crop_size"] = [1024, 1024]

    hypar["model"] = ISNetDIS()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    net = build_model(hypar, device)

    input_path = "./demo/anime-girl-2.jpg"  # Your dataset path
    output_path = "./demo/anime-girl-2-out.png"  # The folder path that you want to save the results

    im_rgba = inference(input_path)
    im_rgba.save(output_path)
