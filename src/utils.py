import zipfile

import cv2
import numpy as np


def get_pixels(mask):
    return np.sum(mask)


def find_bbox(mask):
    contour, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contour:
        return None
    x0, y0 = np.min(np.concatenate(contour), axis=(0, 1))
    x1, y1 = np.max(np.concatenate(contour), axis=(0, 1))
    return x0, y0, x1, y1


def imdecode(im, flags=-1):
    if isinstance(im, zipfile.ZipExtFile):
        im = im.read()
    im = np.asarray(bytearray(im), dtype='uint8')
    return cv2.imdecode(im, flags)


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


def imshow(winname, mat):
    if mat.ndim > 3:
        mat = mat[0]
    if mat.dtype != 'uint8':
        mat = (mat * 255).astype('uint8')
    cv2.imshow(winname, mat)


def get_object(image, bbox, size=512, q=1.00):
    x, y, w, h = corner2center(bbox)
    scaling = 255 / ((w + q * (w + h)) * (h + q * (w + h))) ** 0.5
    mapping = np.array([[scaling, 0, (-x * scaling + size / 2)],
                        [0, scaling, (-y * scaling + size / 2)]], dtype='float')
    crop = cv2.warpAffine(image, mapping, (size, size))
    return crop


def preprocess_input(im):
    x = np.array(im, dtype='float32')
    x /= 255
    x.shape = (1,) + x.shape
    return x


if __name__ == '__main__':
    mask = cv2.imread('00020.png', 0)
    im = cv2.imread('00020.jpg')
    rect = find_bbox(mask)
    # x, y, _, _ = corner2center(rect)
    # x, y = get_rect_sub_pix(im, im, rect)
    # cv2.imshow('s', im)
    # im = get_rect_sub_pix_ex(im, (512, 512), (x, y))
    im = get_object(im, rect)
    mask = get_object(mask, rect)
    # x = warp_affine(im, rect, )
    cv2.imshow('im', im)
    cv2.imshow('mask', mask)
    cv2.waitKey()
