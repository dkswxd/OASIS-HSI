import numpy as np
import os
import cv2

# src = '../data/hsi_256/images/refined/'
src = './svgg/svgg50000/image/'
dst = './pngoasis/'
channel_save = [6, 16, 26]

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
    #     'Require ENVI data: x start = 0'
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
    elif hdr['interleave'].lower() == 'bip':
        img_bytes = img_bytes.reshape((height, width, bands))
        img_bytes = np.transpose(img_bytes, (2,0,1))
    else:
        raise ValueError('Unrecognized interleave, for more information please email:1395133179@qq.com')
    return img_bytes


try:
    os.mkdir(dst)
except FileExistsError:
    pass

for filename in os.listdir(src):
    if filename.endswith('hdr'):
        data = np.transpose(HSIopen(src+filename), (1, 2, 0))
        data = data[:, :, channel_save].astype(np.float32)
        data -= np.mean(data,axis=(0,1),keepdims=True)
        data /= np.std(data,axis=(0,1),keepdims=True)
        data *= 16
        data += 128
        data = data.astype(np.uint8)
        cv2.imwrite(dst+filename.replace('hdr', 'png'), data)




