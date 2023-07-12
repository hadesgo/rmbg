import os
import time
import numpy as np
from skimage import io
import time
from glob import glob

import torch, gc
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from torchvision.transforms.functional import normalize

from models import *

if __name__ == "__main__":
    input_path = "./demo/anime-girl-2.jpg"  # Your dataset path
    model_path = "./saved_models/IS-Net/isnet-general-use.pth"  # the model path
    output_path = "./demo/anime-girl-2-out.png"  # The folder path that you want to save the results
    input_size = [1024, 1024]
    net = ISNetDIS()

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_path))
        net = net.cuda()
    else:
        net.load_state_dict(torch.load(model_path, map_location="cpu"))
    net.eval()
    with torch.no_grad():
        print("im_path: ", input_path)
        im = io.imread(input_path)
        if len(im.shape) < 3:
            im = im[:, :, np.newaxis]
        im_shp = im.shape[0:2]
        im_tensor = torch.tensor(im, dtype=torch.float32).permute(2, 0, 1)
        im_tensor = F.upsample(
            torch.unsqueeze(im_tensor, 0), input_size, mode="bilinear"
        ).type(torch.uint8)
        image = torch.divide(im_tensor, 255.0)
        image = normalize(image, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])

        if torch.cuda.is_available():
            image = image.cuda()
        result = net(image)
        result = torch.squeeze(F.upsample(result[0][0], im_shp, mode="bilinear"), 0)
        ma = torch.max(result)
        mi = torch.min(result)
        result = (result - mi) / (ma - mi)
        io.imsave(
            output_path,
            (result * 255).permute(1, 2, 0).cpu().data.numpy().astype(np.uint8),
        )
