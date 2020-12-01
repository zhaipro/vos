import json
import os
import sys

import cv2
import numpy as np
from tensorflow.keras.models import Model
from tensorflow import keras

import utils
import siammask


def depreprocess(masks):
    masks = np.clip(masks, -10, 10)
    masks = 1 / (1 + np.exp(-masks))

    masks.shape = -1, 127, 127
    result = np.zeros((255, 255))
    w = np.zeros((255, 255))
    for i in range(17 * 17):
        x, y = i % 17, i // 17
        result[y * 8:y * 8 + 127, x * 8:x * 8 + 127] += masks[i]
        w[y * 8:y * 8 + 127, x * 8:x * 8 + 127] += 1
    result /= w
    return result


def main():
    fn = 'tennis/00000.jpg'
    rect = 298, 107, 298 + 164, 107 + 256
    im = cv2.imread(fn)
    template, scaling = utils.get_object(im, rect, size=127)
    cv2.imwrite('template.jpg', template)
    template = utils.preprocess_input(template)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    h, w, _ = im.shape
    writer = cv2.VideoWriter('result.mp4', fourcc, 25, (w, h))

    model = keras.models.load_model('weights.077.h5',
        {'DepthwiseConv2D': siammask.DepthwiseConv2D, 'Reshape': siammask.Reshape},
        compile=False)

    try:
        for i in range(1, 70):
            fn = f'tennis/{i:05}.jpg'
            im = cv2.imread(fn)
            print(fn, im.shape)
            search, scaling = utils.get_object(im, rect, size=255)
            _search = search
            # cv2.imwrite(f'search_{i}.jpg', search)
            search = utils.preprocess_input(search)
            masks = model.predict([template, search])
            masks = depreprocess(masks)
            # cv2.imwrite(f'masks_{i}.jpg', (masks * 255).astype('uint8'))
            # masks = (masks * 255).astype('uint8')
            print('scaling:', scaling)
            tx1, ty1, tx2, ty2 = utils.find_bbox((masks > 0.2).astype('uint8'))
            # cv2.imwrite(f'_search_{i}.jpg', _search[ty1:ty2, tx1:tx2])
            scaling = 1 / scaling
            masks = cv2.resize(masks, None, fx=scaling, fy=scaling)
            print('masks.shape', masks.shape)
            cx, cy, _, _ = utils.corner2center(rect)
            hh, ww = masks.shape
            x1, y1, x2, y2 = utils.center2corner((cx, cy, ww, hh))
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            print('tx:',tx1, ty1, tx2, ty2)
            tx1 *= scaling
            ty1 *= scaling
            tx2 *= scaling
            ty2 *= scaling
            tx1 += cx - scaling * 255 / 2
            ty1 += cy - scaling * 255 / 2
            tx2 += cx - scaling * 255 / 2
            ty2 += cy - scaling * 255 / 2
            print('prect:', rect, cx, cy)
            rect = tx1, ty1, tx2, ty2
            print('nrect:', rect, cx, cy)

            if x1 < 0:
                masks = masks[:, -x1:]
                x1 = 0
            if y1 < 0:
                masks = masks[-y1:]
                y1 = 0
            if x2 > w:
                x2 = w
            masks = masks[:, :x2 - x1]
            if y2 > h:
                y2 = h
            masks = masks[:y2 - y1]
            # x1 = max(0, x1)
            # y1 = max(0, y1)
            # x2 = min(w, x2)
            # y2 = min(h, y2)
            print('aaa:', x1, y1, x2, y2, masks.shape)
            im[int(y1):int(y2), int(x1):int(x2), 2] = im[int(y1):int(y2), int(x1):int(x2), 2] * (1 - masks) + 255 * masks
            writer.write(im)
            # exit()
    except:
        pass

    writer.release()


if __name__ == '__main__':
    main()
