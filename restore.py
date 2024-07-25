import os
import glob
import torch
import torch.nn.functional as F
from typing import List, Tuple
import numpy as np
from RT4KSR.code.utils.metrics import calculate_psnr, calculate_ssim
import sys
sys.path.append("RT4KSR/code")
if "RT4KSR/code" in sys.path:
    import model


def load_checkpoint(model, device, path):
    checkpoint = glob.glob(path)
    if isinstance(checkpoint, List):
        checkpoint = checkpoint.pop(0)
    checkpoint = torch.load(checkpoint, map_location=device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model


class Restoration:
    def __init__(self, path, config):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = torch.nn.DataParallel(
            model.__dict__[config.arch](config)
        ).to(device)
        net = load_checkpoint(net, device, path)
        net.eval()
        self.net = net

    def restore(self, distorted, reference):
        with torch.no_grad():
            out = self.net((reference.unsqueeze(0), distorted.unsqueeze(0)))
            return out.squeeze(0)


def get_restoration(path, args):
    args.__setattr__("scale", 1)
    args.__setattr__("arch", "nerfsrresnet")
    args.__setattr__("feature_channels", 24)
    args.__setattr__("num_blocks", 4)
    return Restoration(path, args)


def tensor2uint(img):
    img = img.data.squeeze().float().clamp(0, 255).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return np.uint8((img).round())


def metrics(out, hr_img):
    out = tensor2uint(out*255.)
    hr_img = tensor2uint(hr_img*255.)
    return dict(
        psnr_rgb=calculate_psnr(out, hr_img, crop_border=0),
        ssim_rgb=calculate_ssim(out, hr_img, crop_border=0),
        psnr_y=calculate_psnr(out, hr_img, crop_border=0, test_y_channel=True),
        ssim_y=calculate_ssim(out, hr_img, crop_border=0, test_y_channel=True),
    )
