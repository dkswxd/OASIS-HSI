import os
import numpy as np
import torch
from torchvision import transforms as TR
import torch.nn.functional as F
from PIL import Image

src = 'hsi'
folders = ['refined', 'unrefined']

def copy_hdr(src_hdr, dst_hdr):
    with open(src_hdr) as fin:
        with open(dst_hdr, 'w') as fout:
            for line in fin.readlines():
                fout.write(line)


for folder in folders:
    src_folder = os.path.join(src, 'images', folder)
    for filename in os.listdir(src_folder):
        if filename.endswith('.hdr'):
            copy_hdr('sample.hdr',os.path.join(src_folder, filename))
