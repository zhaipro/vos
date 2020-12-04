import os

import cv2
import numpy as np


os.makedirs('mm_masks', exist_ok=True)

data = np.load('result.npz')
masks = data['masks']

# masks = np.clip(masks, -10, 10)
# masks = (masks > 0.3).astype('float32')
# masks = 1 / (1 + np.exp(-masks))

# cv2.imwrite('a.jpg', masks[0] * 255)
# exit()

masks.shape = -1, 127, 127
for i, mask in enumerate(masks):
    cv2.imwrite(f'mm_masks/{i // 17}_{i % 17}.jpg', (mask + 0) * 255)

# result = np.ones((255, 255))
result = np.zeros((255, 255))
w = np.zeros((255, 255))
for i in range(17 * 17):
    x, y = i % 17, i // 17
    # r = result[y * 8:y * 8 + 127, x * 8:x * 8 + 127]
    # r[:] = np.maximum(r, masks[i])
    # result[y * 8:y * 8 + 127, x * 8:x * 8 + 127] *= masks[i]
    # _w = np.abs(masks[i] - 0.5) * 2
    # result[y * 8:y * 8 + 127, x * 8:x * 8 + 127] += _w * masks[i]
    result[y * 8:y * 8 + 127, x * 8:x * 8 + 127] += masks[i]
    w[y * 8:y * 8 + 127, x * 8:x * 8 + 127] += 1
    # w[y * 8:y * 8 + 127, x * 8:x * 8 + 127] += _w
result /= w
cv2.imwrite(f'mm_masks/_rr.jpg', result * 255)
