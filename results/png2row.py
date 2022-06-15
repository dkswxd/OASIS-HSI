import os
import numpy as np
import cv2
from PIL import Image

label_src = '../data/hsi_256/annotations/refined'
label_dst = './label'

src_folders = ['./pngraw/', './label/', './png10000/', './png233333/', './png666666/']
row_dst = './row'
height = 256
width = 320
step = 5

try:
    os.mkdir(label_dst)
except FileExistsError:
    pass

try:
    os.mkdir(row_dst)
except FileExistsError:
    pass

for filename in os.listdir(label_src):
    img = cv2.imread(os.path.join(label_src, filename))
    img[img >= 1] = 255
    cv2.imwrite(os.path.join(label_dst, filename), img)

for filename in os.listdir(src_folders[-1]):
    row = np.ones((height, (width+step)*len(src_folders), 3), dtype=np.uint8) * 255
    x = 0
    for src_folder in src_folders:
        img = cv2.imread(os.path.join(src_folder, filename))
        row[:, x:x+width, :] = img
        x += width+step
    cv2.imwrite(os.path.join(row_dst, filename), row)

