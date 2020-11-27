import cv2
import numpy as np


d = np.load('datasets.npz')
scores = d['scores']
masks = d['masks']
scores.shape = 17, 17
print(scores)
print(masks.shape)
masks = (masks + 1) * 127.5
masks.shape = -1, 127, 127
scores.shape = -1
for i, (mask, score) in enumerate(zip(masks, scores)):
	if score > 0.9:
		score = 't'
	elif score < -0.9:
		score = 'f'
	else:
		score = 'n'
	cv2.imwrite(f'masks/{i // 17}x{i % 17}_{score}.jpg', mask)
