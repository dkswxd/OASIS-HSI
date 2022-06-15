import os
import numpy as np
import torch
from torchvision import transforms as TR
import torch.nn.functional as F
from PIL import Image

src = 'hsi'
dst = 'hsi_256'
folders = ['refined', 'unrefined']
height, width = (256, 320)

def HSIopen(filename):
    ENVI_data_type = [None,
                      np.uint8,  # 1
                      np.int16,  # 2
                      np.int32,  # 3
                      np.float32,  # 4
                      np.float64,  # 5
                      None,
                      None,
                      None,
                      None,
                      None,
                      None,
                      np.uint16,  # 12
                      np.uint32, ]  # 13
    hdr = dict()
    with open(filename) as f:
        for line in f.readlines():
            if '=' not in line:
                continue
            else:
                key, value = line.split('=')
                key = key.strip()
                value = value.strip()
                hdr[key] = value

    assert int(hdr['data type']) <= len(ENVI_data_type) and ENVI_data_type[int(hdr['data type'])] != None, \
        'Unrecognized data type'

    data_type = int(hdr['data type'])
    header_offset = int(hdr['header offset'])
    height = int(hdr['lines'])
    width = int(hdr['samples'])
    bands = int(hdr['bands'])
    img_bytes = np.fromfile(filename.replace('.hdr', '.raw'),
                            dtype=ENVI_data_type[data_type],
                            offset=header_offset)
    if hdr['interleave'].lower() == 'bsq':
        img_bytes = img_bytes.reshape((bands, height, width))
    else:
        raise ValueError('Unrecognized interleave, for more information please email:1395133179@qq.com')
    return img_bytes

def torch_resize(image, size, mode):
    with torch.no_grad():
        image = torch.tensor(image.astype(np.float32)).unsqueeze(0)
        image = F.interpolate(image, size=size, mode=mode)
        image = image.squeeze().numpy().astype(np.uint16)
    return image
def copy_hdr(src_hdr, dst_hdr):
    with open(src_hdr) as fin:
        with open(dst_hdr, 'w') as fout:
            for line in fin.readlines():
                fout.write(line.replace('1024', '256').replace('1280', '320'))


for folder in folders:
    src_folder = os.path.join(src, 'images', folder)
    dst_folder = os.path.join(dst, 'images', folder)
    src_label_folder = os.path.join(src, 'annotations', folder)
    dst_label_folder = os.path.join(dst, 'annotations', folder)
    for filename in os.listdir(src_folder):
        if filename.endswith('.hdr'):
            image = HSIopen(os.path.join(src_folder, filename))
            image = torch_resize(image, (height, width), 'bicubic')
            image.tofile(os.path.join(dst_folder, filename.replace('.hdr', '.raw')))
            copy_hdr(os.path.join(src_folder, filename), os.path.join(dst_folder, filename))
            label = Image.open(os.path.join(src_label_folder, filename.replace('.hdr', '.png')))
            label = label.resize((width, height), Image.NEAREST)
            label.save(os.path.join(dst_label_folder, filename.replace('.hdr', '.png')))
