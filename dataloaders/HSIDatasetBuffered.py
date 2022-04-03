import random
import torch
from torchvision import transforms as TR
import os
from PIL import Image
import numpy as np
import torch.nn.functional as F

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
    # assert hdr['file type'] == 'ENVI Standard', \
    #     'Require ENVI data: file type = ENVI Standard'
    # assert hdr['byte order'] == '0', \
    #     'Require ENVI data: byte order = 0'
    # assert hdr['x start'] == '0', \
    #     '
    #
    #
    #     ENVI data: x start = 0'
    # assert hdr['y start'] == '0', \
    #     'Require ENVI data: y start = 0'
    assert int(hdr['data type']) <= len(ENVI_data_type) and ENVI_data_type[int(hdr['data type'])] != None, \
        'Unrecognized data type'

    data_type = int(hdr['data type'])
    header_offset = int(hdr['header offset'])
    height = int(hdr['lines'])
    width = int(hdr['samples'])
    bands = int(hdr['bands'])
    img_bytes = np.fromfile(filename.replace('.hdr', '.raw'), dtype=ENVI_data_type[data_type],
                            offset=header_offset)
    if hdr['interleave'].lower() == 'bsq':
        img_bytes = img_bytes.reshape((bands, height, width))
    else:
        raise ValueError('Unrecognized interleave, for more information please email:1395133179@qq.com')
    return img_bytes[range(4,36), :, :]



class HSIDatasetBuffered(torch.utils.data.Dataset):
    def __init__(self, opt, for_metrics):
        # opt.load_size = (512, 640)
        # opt.crop_size = 448
        # opt.load_size = (256, 320)
        # opt.crop_size = 256
        opt.load_size = (128, 160)
        opt.crop_size = 128
        # opt.load_size = (64, 80)
        # opt.crop_size = 64
        opt.label_nc = 2
        opt.contain_dontcare_label = False
        opt.semantic_nc = 2 # label_nc + unknown
        opt.cache_filelist_read = False
        opt.cache_filelist_write = False
        opt.aspect_ratio = 1.0

        self.opt = opt
        self.for_metrics = for_metrics
        self.images, self.labels, self.paths, self.names = self.list_images()

    def __len__(self,):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        image, label = self.transforms(image, label)
        label = label * 255
        return {"image": image, "label": label, "name": self.names[0][idx]}

    def list_images(self):
        mode = "refined" if self.opt.phase == "test" or self.for_metrics else "unrefined"
        # mode = "unrefined" if self.opt.phase == "test" or self.for_metrics else "unrefined"
        path_img = os.path.join(self.opt.dataroot, "images", mode)
        path_lab = os.path.join(self.opt.dataroot, "annotations", mode)
        img_list = os.listdir(path_img)
        lab_list = os.listdir(path_lab)
        img_list = [filename for filename in img_list if ".hdr" in filename or ".jpg" in filename]
        lab_list = [filename for filename in lab_list if ".png" in filename or ".jpg" in filename]
        images = sorted(img_list)
        labels = sorted(lab_list)
        assert len(images)  == len(labels), "different len of images and labels %s - %s" % (len(images), len(labels))
        for i in range(len(images)):
            assert os.path.splitext(images[i])[0] in os.path.splitext(labels[i])[0], '%s and %s are not matching' % (images[i], labels[i])
        imagesx = [HSIopen(os.path.join(path_img, name)) for name in images]
        labelsx = [Image.open(os.path.join(path_lab, name)) for name in labels]
        return imagesx, labelsx, (path_img, path_lab), (images, labels)

    def torch_resize(self, image, size, mode):
        with torch.no_grad():
            image = torch.tensor(image.astype(np.float32)).cuda().unsqueeze(0)
            image = F.interpolate(image, size=size, mode=mode)
            image = image.squeeze().cpu().numpy().astype(np.uint16)
        return image

    def transforms(self, image, label):
        assert image.shape[1:] == label.size[::-1]
        new_height, new_width = self.opt.load_size
        # new_width, new_height = self.opt.load_size

        image = self.torch_resize(image, (new_height, new_width), 'bicubic')
        image = np.transpose(image, (1, 2, 0))
        label = np.array(TR.functional.resize(label, (new_height, new_width), Image.NEAREST))
        # crop
        if not (self.opt.phase == "test" or self.opt.no_flip or self.for_metrics):
            crop_x = random.randint(0, np.maximum(0, new_width  - self.opt.crop_size))
            crop_y = random.randint(0, np.maximum(0, new_height - self.opt.crop_size))
            # image = image.crop((crop_x, crop_y, crop_x + self.opt.crop_size, crop_y + self.opt.crop_size))
            # label = label.crop((crop_x, crop_y, crop_x + self.opt.crop_size, crop_y + self.opt.crop_size))
            image = image[crop_y : crop_y + self.opt.crop_size, crop_x : crop_x + self.opt.crop_size, :]
            label = label[crop_y : crop_y + self.opt.crop_size, crop_x : crop_x + self.opt.crop_size]
        # flip
        if not (self.opt.phase == "test" or self.opt.no_flip or self.for_metrics):
            if random.random() < 0.5:
                # image = TR.functional.hflip(image)
                # label = TR.functional.hflip(label)
                image = image[:, ::-1, :]
                label = label[:, ::-1]
            if random.random() < 0.5:
                # image = TR.functional.vflip(image)
                # label = TR.functional.vflip(label)
                image = image[::-1, :, :]
                label = label[::-1, :]
        # normalize
        image = image.astype(np.float32)
        image -= np.mean(image,axis=(0,1),keepdims=True)
        image /= (np.clip(np.std(image,axis=(0,1),keepdims=True), 1e-6, 1e6) * 4)
        # to tensor
        image = TR.functional.to_tensor(image.copy())
        label = TR.functional.to_tensor(label.copy())
        return image, label
