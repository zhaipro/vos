import argparse
import h5py
import json
import os
import scipy.misc
import sys
import zipfile
import numpy as np
import cv2
from os.path import join


def get_pixels(mask):
    return np.sum(mask)


def find_bbox(mask):
    contour, _ = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    x0, y0 = np.min(np.concatenate(contour), axis=(0, 1))
    x1, y1 = np.max(np.concatenate(contour), axis=(0, 1))
    return x0, y0, x1, y1


def imdecode(im, flags=-1):
    if isinstance(im, zipfile.ZipExtFile):
        im = im.read()
    im = np.asarray(bytearray(im), dtype='uint8')
    return cv2.imdecode(im, flags)


def convert_ytb_vos(data_dir, out_dir):
    json_name = 'instances_%s.json'
    num_obj = 0
    num_ann = 0

    print('Starting')
    ann_dict = {}
    ann_dir = 'valid/Annotations/'

    zf = zipfile.ZipFile(data_dir)
    with zf.open('valid/meta.json') as fp:
        json_ann = json.load(fp)

    for vid, video in enumerate(json_ann['videos']):
        v = json_ann['videos'][video]
        frames = []
        for obj in v['objects']:
            o = v['objects'][obj]
            frames.extend(o['frames'])
        frames = sorted(set(frames))[:1]

        annotations = []
        instanceIds = []
        for frame in frames:
            file_name = f'{video}/{frame}'
            fullname = f'{ann_dir}{file_name}.png'

            with zf.open(fullname) as fp:
                img = imdecode(fp, flags=0)

            h, w = img.shape[:2]

            objects = dict()
            for instanceId in np.unique(img):
                if instanceId == 0:
                    continue
                instanceObj = Instance(img, instanceId)
                instanceObj_dict = instanceObj.toDict()
                mask = (img == instanceId).astype(np.uint8)
                contour, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                print(len(contour[0]), type(contour[0]), contour[0].shape)
                polygons = [c.reshape(-1).tolist() for c in contour]
                instanceObj_dict['contours'] = [p for p in polygons if len(p) > 4]
                if len(instanceObj_dict['contours']) and instanceObj_dict['pixelCount'] > 1000:
                    objects[instanceId] = instanceObj_dict

            for objId in objects:
                if len(objects[objId]) == 0:
                    continue
                obj = objects[objId]
                len_p = [len(p) for p in obj['contours']]
                if min(len_p) <= 4:
                    print('Warning: invalid contours.')
                    continue  # skip non-instance categories

                ann = dict()
                ann['h'] = h
                ann['w'] = w
                ann['file_name'] = file_name
                ann['id'] = int(objId)
                ann['area'] = obj['pixelCount']
                ann['bbox'] = xyxy_to_xywh(polys_to_boxes([obj['contours']])).tolist()[0]

                annotations.append(ann)
                instanceIds.append(objId)
                num_ann += 1
        instanceIds = sorted(set(instanceIds))
        num_obj += len(instanceIds)
        video_ann = {str(iId): [] for iId in instanceIds}
        for ann in annotations:
            video_ann[str(ann['id'])].append(ann)

        ann_dict[video] = video_ann
        if vid % 50 == 0 and vid != 0:
            print("process: %d video" % (vid+1))

    print("Num Videos: %d" % len(ann_dict))
    print("Num Objects: %d" % num_obj)
    print("Num Annotations: %d" % num_ann)

    items = list(ann_dict.items())
    train_dict = dict(items)
    with open(os.path.join(out_dir, json_name % 'train'), 'w') as outfile:
        json.dump(train_dict, outfile, indent=2)


if __name__ == '__main__':
    im = cv2.imread('00020.png', 0)
    rect = find_bbox(im)
    print(rect)
    exit()
    if len(sys.argv) < 3:
        print('python preprocess.py <datadir> <outdir>')
        exit(1)
    datadir, outdir = sys.argv[1], sys.argv[2]
    convert_ytb_vos(datadir, outdir)
