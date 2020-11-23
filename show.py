import os

import cv2
import numpy as np


os.makedirs('mm_masks', exist_ok=True)

masks = np.load('masks.npy')
# masks = np.tanh(masks)
masks = 1 / (1 + np.exp(-masks))
masks.shape = -1, 127, 127
for i, mask in enumerate(masks):
    cv2.imwrite(f'mm_masks/{i // 17}_{i % 17}.jpg', (mask + 0) * 255)

result = np.zeros((255, 255))
w = np.zeros((255, 255))
for i in masks.sum((1, 2)).argsort()[-5:]:
    x, y = i % 17, i // 17
    result[y * 8:y * 8 + 127, x * 8:x * 8 + 127] += masks[i]
    w[y * 8:y * 8 + 127, x * 8:x * 8 + 127] += 1
result /= w
cv2.imwrite(f'mm_masks/_rr.jpg', result * 255)
