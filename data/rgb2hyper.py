

'''
高光谱标签转RGB标签说明:
输入文件夹写进 sources，每个文件内都是xxx.jpg和xxx.raw
输出文件夹写进 targets
首先is_testing = True，在image和label中找4个对应的点，把鼠标放在点上窗口下面会有坐标
rgb的点放进数组 src_points
hyper的点放进数组 dst_points
设定is_testing = False，开始转换

'''

from __future__ import division
import os
import numpy as np
import cv2
import sys

def open_image(filename):
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
        img_bytes = np.transpose(img_bytes, (1,2,0))
    else:
        raise ValueError('Unrecognized interleave, for more information please email:1395133179@qq.com')
    return img_bytes

def callback(object):
    pass

is_testing = True

sources = ['./hsi_256/images/refined/','./hsi_256/images/unrefined/']
targets = ['./hsirgb/images/refined/','./hsirgb/images/unrefined/']

for source, target in zip(sources, targets):
    for filename in os.listdir(source):
        if filename.endswith('hdr'):
            rgb = cv2.imread(os.path.join(source,filename).replace('-L.hdr', '.jpg'))
            assert rgb is not None
            raw = open_image(os.path.join(source,filename))[:,:,10:40:10] >> 6
            raw = raw.astype(np.uint8)
            
            if is_testing:
                # raw = cv2.resize(raw,(raw.shape[1] // 4, raw.shape[0] // 4))
                src_points = np.float32([[62,382],[242,74],[434,346],[412,116]])*4
                dst_points = np.float32([[20,20],[120,231],[258,65],[234,215]])*4
                affine = cv2.getPerspectiveTransform(src_points, dst_points)
                affineed = cv2.warpPerspective(rgb,affine,(raw.shape[1]*4, raw.shape[0]*4))
                rgb = cv2.resize(rgb,(rgb.shape[1] // 4, rgb.shape[0] // 4))[::-1,:,:]
                affineed = cv2.resize(affineed,(affineed.shape[1] // 4, affineed.shape[0] // 4))
                cv2.imshow('rgb', rgb)
                cv2.imshow('hyper', raw)
                cv2.imshow('affineed', affineed)
                cv2.waitKey()
            else:
                src_points = np.float32([[62,382],[242,74],[434,346],[412,116]])*4
                dst_points = np.float32([[20,20],[120,231],[258,65],[234,215]])*4
                affine = cv2.getPerspectiveTransform(src_points, dst_points)
                rgb = cv2.warpPerspective(rgb,affine,(raw.shape[1]*4, raw.shape[0]*4))
                cv2.imwrite(os.path.join(target,filename).replace('-L.hdr', '.jpg'),rgb)


            # cv2.namedWindow('masked')
            # cv2.createTrackbar('11','masked', 1, 1000, callback)
            # cv2.createTrackbar('12','masked', 1, 1000, callback)
            # cv2.createTrackbar('13','masked', 1, 1000, callback)
            # cv2.createTrackbar('22','masked', 1, 1000, callback)
            # cv2.createTrackbar('21','masked', 1, 1000, callback)
            # cv2.createTrackbar('23','masked', 1, 1000, callback)
            # img = cv2.imread(source+file[:-11]+'.jpg')
            # label = cv2.imread(source+file, -1)
            # label = cv2.resize(label, (shp[1], shp[0]))
            # while (True):
            #     affine[0,0] = cv2.getTrackbarPos('11','masked') / 5000 + 0.8
            #     affine[0,1] = cv2.getTrackbarPos('12','masked') / 5000 - 0.1
            #     affine[0,2] = cv2.getTrackbarPos('13','masked') / 20 + 180
            #     affine[1,0] = cv2.getTrackbarPos('21','masked') / 5000 - 0.1
            #     affine[1,1] = cv2.getTrackbarPos('22','masked') / 5000 + 0.8
            #     affine[1,2] = cv2.getTrackbarPos('23','masked') / 20 + 80
            #     label_ = cv2.warpAffine(label,affine,(shp[1], shp[0]))
            #     print(affine)
            #     img2show = img.copy()
            #     img2show[:,:,1] += label_ // 4
            #     img2show = cv2.resize(img2show,(shp[1] // 3, shp[0] // 3))
            #     cv2.imshow('masked', img2show)
            #     cv2.waitKey(1000)
    # break








