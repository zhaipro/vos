import json
import os
import sys
import zipfile
import collections

import cv2
import numpy as np

import utils


def func(fn='train.zip'):
    results = []

    zf = zipfile.ZipFile(fn)
    with zf.open('train/meta.json') as fp:
        meta = json.load(fp)

    for path, data in meta['videos'].items():
        frames = set()
        for obj in data['objects'].values():
            frames.update(obj['frames'])
        frames = sorted(frames)

        objects = collections.defaultdict(list)
        for frame in frames:
            ofn = f'train/Annotations/{path}/{frame}.png'
            with zf.open(ofn) as fp:
                mask = utils.imdecode(fp, 0)
            for c in np.unique(mask):
                if c == 0:
                    continue
                objects[int(c)].append(frame)

        for color, fns in objects.items():
            results.append({'path': path, 'color': color, 'fns': fns})
    with open('meta.json', 'w') as fp:
        json.dump(results, fp, indent=2)

if __name__ == '__main__':
    func()
