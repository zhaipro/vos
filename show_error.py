import cv2
import numpy as np



data = np.load('errors.npz')
im, mask = data['im'], data['mask']

cv2.imwrite('_mask.jpg', mask * 255)
cv2.imwrite('_im.jpg', im)


def preprocess_mask(mask):
    # h, w = mask.shape
    ms, ss = 0, 0
    mask_weight = np.zeros(17 * 17, dtype='float32')
    score_weight = np.zeros(17 * 17, dtype='float32')
    weight = mask.sum()
    masks = np.zeros((17 * 17, 127, 127), dtype='float32')
    scores = np.zeros(17 * 17)
    for i in range(17 * 17):
        x = i % 17
        y = i // 17
        m = mask[y * 8:y * 8 + 127, x * 8:x * 8 + 127]
        score = m.sum() / weight
        print(x, y, score)
        masks[i] = (m - 0.5) * 2
        if score > 0.99:
            scores[i] = 1
            score_weight[i] = 1
            mask_weight[i] = 1
            ss += 1
            ms += 1
            # print(a, weight, outputs[i].min(), outputs[i].max())
        elif score < 0.75:
            score_weight[i] = 1
            # masks[i] = -1
            scores[i] = -1
            ss += 1
        else:
            # masks[i] = m - 1
            scores[i] = 0
    mask_weight.shape = 1, 17, 17
    score_weight.shape = 1, 17, 17
    mask_weight *= 17 * 17 / ms
    score_weight *= 17 * 17 / ss
    return masks, scores, (mask_weight, score_weight)


def find_bbox(mask):
    contour, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contour:
        return None
    x0, y0 = np.min(np.concatenate(contour), axis=(0, 1))
    x1, y1 = np.max(np.concatenate(contour), axis=(0, 1))
    return x0, y0, x1, y1

def corner2center(corner):
    x1, y1, x2, y2 = corner
    x, y = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    w, h = (x2 - x1), (y2 - y1)
    return x, y, w, h


def center2corner(center):
    x, y, w, h = center
    x1, y1 = x - w * 0.5, y - h * 0.5
    x2, y2 = x + w * 0.5, y + h * 0.5
    return x1, y1, x2, y2

def get_object(image, bbox, size=255, q=0.50, move=(0, 0), flip=False, border=0):
    x, y, w, h = corner2center(bbox)
    scaling = 127 / ((w + q * (w + h)) * (h + q * (w + h))) ** 0.5
    x_scaling = scaling
    y_scaling = scaling
    mx, my = move
    mx = -x * scaling + size / 2 + mx
    my = -y * scaling + size / 2 + my
    if flip:
        mx += size
        x_scaling = -x_scaling
    mapping = np.array([[x_scaling, 0, mx],
                        [0, y_scaling, my]], dtype='float')
    crop = cv2.warpAffine(image, mapping, (size, size), borderValue=border)
    return crop


# preprocess_mask(mask)
bbox = find_bbox(mask.astype('uint8'))
crop = get_object(mask, bbox, border=0)
print(bbox, crop.shape)
cv2.imshow('a', crop * 255)
cv2.waitKey()
print(crop.shape, crop.dtype)
preprocess_mask(crop)
exit()

masks = (masks + 1) * 127.5
masks.shape = -1, 127, 127
cv2.imwrite('errors/mask.jpg', mask * 255)
for i, mask in enumerate(masks):
    cv2.imwrite(f'errors/{i // 17}x{i % 17}.jpg', mask)
