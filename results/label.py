import numpy as np
import os
import cv2

src = './svgg/refined/label/'

for filename in os.listdir(src):
    img = cv2.imread(os.path.join(src, filename), -1)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    print(np.unique(img))
    img[img<=70] = 0
    img[img>70] = 1
    cv2.imwrite(os.path.join(src, filename),img)




