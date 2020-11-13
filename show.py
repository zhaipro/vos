import cv2
import numpy as np


masks = np.load('masks.npy')
masks = 1 / (1 + np.exp(-masks))
masks.shape = -1, 63, 63
for i, mask in enumerate(masks):
    cv2.imwrite(f'mm_masks/{i // 17}_{i % 17}.jpg', mask * 255)
i = masks.sum((1, 2)).argmax()

result = np.zeros((127, 127))
w = np.zeros((127, 127))
for i, mask in enumerate(masks):
    x, y = i % 17, i // 17
    result[y * 4:y * 4 + 63, x * 4:x * 4 + 63] += masks[i]
    w[y * 4:y * 4 + 63, x * 4:x * 4 + 63] += 1
result /= w
cv2.imwrite(f'mm_masks/_rr.jpg', result * 255)
