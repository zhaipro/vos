import json
import os
import sys
import zipfile

import cv2
import numpy as np
from tensorflow import keras
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

import unet
import utils


class Dataset:

    def __init__(self, fn='train.zip'):
        self.zf = zipfile.ZipFile(fn)
        with self.zf.open('train/meta.json') as fp:
            self.meta = json.load(fp)

    def preprocess_inputs(self, mask):
        mask = (mask == self.colors).all(axis=2)
        mask.dtype = 'uint8'
        return mask

    def _generator(self):
        for path, data in self.meta['videos'].items():
            frames = set()
            for obj in data['objects'].values():
                frames.update(obj['frames'])
            for frame in frames:
                ifn = f'train/JPEGImages/{path}/{frame}.jpg'
                ofn = f'train/Annotations/{path}/{frame}.png'
                with self.zf.open(ifn) as fp:
                    im = utils.imdecode(fp)
                with self.zf.open(ofn) as fp:
                    mask = utils.imdecode(fp, 0)
                for c in np.unique(mask):
                    if c == 0:
                        continue
                    y = mask == c
                    y.dtype = 'uint8'
                    bbox = utils.find_bbox(y)
                    if not bbox:
                        continue
                    x = utils.get_object(im, bbox)
                    y = utils.get_object(y, bbox)
                    x.shape = (1,) + x.shape
                    y.shape = (1,) + y.shape + (1,)
                    yield x.astype('float32'), y.astype('float32')

    def _generator(self):
        while True:
            yield from self.generator()


def mlearn():
    version = '1.0.0'
    dataset = Dataset()
    xy_train = dataset.generator()
    xy_test = dataset.generator()
    model = unet.get_unet_256()
    # model = keras.models.load_model('weights.lip.2.0.7.h5', compile=False)
    reduce_lr = ReduceLROnPlateau(patience=3, verbose=1)
    mcp = ModelCheckpoint(filepath='weights.{epoch:03d}.h5')
    model.compile(optimizer=keras.optimizers.RMSprop(lr=0.0001),
                  loss=unet.bce_dice_loss,
                  metrics=[unet.dice_coeff, unet.psnr])
    model.fit_generator(xy_train,
                        steps_per_epoch=29500,
                        epochs=20,
                        validation_data=xy_test,
                        validation_steps=500,
                        callbacks=[reduce_lr, mcp])
    model.save(f'weights.{version}.h5', include_optimizer=False)
    result = model.evaluate_generator(xy_test, steps=500)
    print(result)
    predict(version)


def main(fn):
    model = keras.models.load_model('weights.001.h5', compile=False)
    im = cv2.imread('00020.jpg')
    mask = cv2.imread('00020.png', 0)

    c = np.unique(mask)[0]
    y = mask == c
    y.dtype = 'uint8'
    bbox = utils.find_bbox(y)
    x = utils.get_object(im, bbox)
    y = utils.get_object(y, bbox)
    x.shape = (1,) + x.shape
    y.shape = (1,) + y.shape + (1,)
    x = x.astype('float32')
    masks = model.predict(x)
    cv2.imwrite('a.jpg', masks[0])

if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        mlearn()
