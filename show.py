import cv2
import numpy as np


masks = np.load('masks.npy')
masks = np.tanh(masks)
# masks = 1 / (1 + np.exp(-masks))
masks.shape = -1, 127, 127
for i, mask in enumerate(masks):
    cv2.imwrite(f'mm_masks/{i // 17}_{i % 17}.jpg', (mask + 1) * 127.5)
i = masks.sum((1, 2)).argmax()
print(i)

result = np.zeros((255, 255))
w = np.zeros((255, 255))
for i, mask in enumerate(masks):
    x, y = i % 17, i // 17
    result[y * 8:y * 8 + 127, x * 8:x * 8 + 127] += (masks[i] + 1) / 2
    w[y * 8:y * 8 + 127, x * 8:x * 8 + 127] += 1
result /= w
cv2.imwrite(f'mm_masks/_rr.jpg', result * 255)
