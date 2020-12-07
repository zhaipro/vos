import json
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tensorflow import keras
from tensorflow.keras.models import Model

import utils
import siammask


def _depreprocess(masks):
    # masks = np.clip(masks, -10, 10)
    # masks = 1 / (1 + np.exp(-masks))

    masks.shape = -1, 127, 127
    result = np.zeros((255, 255))
    w = np.zeros((255, 255))
    for i in range(17 * 17):
        x, y = i % 17, i // 17
        result[y * 8:y * 8 + 127, x * 8:x * 8 + 127] += masks[i]
        w[y * 8:y * 8 + 127, x * 8:x * 8 + 127] += 1
    result /= w
    return result


def depreprocess_2(masks, scores):
    masks.shape = -1, 127, 127
    scores.shape = -1
    result = np.zeros((255, 255))
    w = np.zeros((255, 255)) + 1e-7
    for i in scores.argsort()[-5:]:
        x, y = i % 17, i // 17
        print('pos:', x, y, scores[i])
        result[y * 8:y * 8 + 127, x * 8:x * 8 + 127] += masks[i]
        w[y * 8:y * 8 + 127, x * 8:x * 8 + 127] += 1
    result /= w
    return result


def depreprocess(masks, scores):
    masks.shape = -1, 127, 127
    result = np.zeros((255, 255))
    x, y = 8, 8
    result[y * 8:y * 8 + 127, x * 8:x * 8 + 127] += masks[144]
    return result


def put_text(img, text, xy, fill=(255, 255, 255), size=20):
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype('SourceHanSansCN-Bold.otf', size, encoding='utf-8')
    draw.text(xy, text, fill, font=font)
    return np.asarray(img)


def main(text):
    os.makedirs('tennis_results', exist_ok=True)
    fn = 'tennis/00000.jpg'
    rect = 298, 107, 298 + 164, 107 + 256
    # rect = 348, 220, 348 + 79, 220 + 28
    im = cv2.imread(fn)
    template, scaling = utils.get_object(im, rect, size=127)
    cv2.imwrite('template.jpg', template)
    template = utils.preprocess_input(template)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    h, w, _ = im.shape
    writer = cv2.VideoWriter('result.mp4', fourcc, 25, (w, h))

    model = keras.models.load_model('weights.100.h5',
        {'DepthwiseConv2D': siammask.DepthwiseConv2D, 'Reshape': siammask.Reshape},
        compile=False)

    try:
        for i in range(1, 70):
            fn = f'tennis/{i:05}.jpg'
            print(fn)
            im = cv2.imread(fn)
            print(fn, im.shape)
            search, scaling = utils.get_object(im, rect, size=255)
            search = utils.preprocess_input(search)
            masks, scores = model.predict([template, search])
            masks = depreprocess(masks, scores)
            cv2.imwrite(f'tennis_results/{i}_masks.jpg', masks * 255)
            cv2.imwrite(f'tennis_results/{i}_search.jpg', search[0] * 255)
            # masks = model.predict([template, search])
            # masks = _depreprocess(masks)
            print('scaling:', scaling)
            tx1, ty1, tx2, ty2 = utils.find_bbox((masks > 0.20).astype('uint8'))
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
            x2 = min(w, x2)
            masks = masks[:, :x2 - x1]
            y2 = min(h, y2)
            masks = masks[:y2 - y1]
            print('aaa:', x1, y1, x2, y2, masks.shape)
            # im[int(y1):int(y2), int(x1):int(x2), 2] = im[int(y1):int(y2), int(x1):int(x2), 2] * (1 - masks) + 255 * masks
            _masks = np.zeros((h, w, 1), dtype='float')
            _masks[y1:y2, x1:x2, 0] = masks
            _masks = (text > 127) * (1 - _masks)
            im[:] = im * (1 - _masks) + text * _masks
            writer.write(im)
            # exit()
    except Exception as e:
        raise e
        print(e)

    writer.release()


if __name__ == '__main__':
    text = put_text(np.zeros((480, 854, 3), dtype='uint8'), '宅教授', (45, 100), (255, 192, 203), size=250)
    main(text)
