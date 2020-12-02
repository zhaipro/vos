# http://www.robots.ox.ac.uk/~luca/siamese-fc.html
import json
import os
import sys
import zipfile
import random

import cv2
import numpy as np
from tensorflow import keras
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D, Input, Activation
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
import tensorflow as tf

import utils
import resnet


def res_down():
    inputs = Input(shape=(None, None, 3))
    rn50 = resnet.ResNet50(input_tensor=inputs)
    p0 = rn50.get_layer('conv1_relu').output
    p1 = rn50.get_layer('conv2_block3_out').output
    p2 = rn50.get_layer('conv3_block4_out').output
    p3 = rn50.get_layer('conv4_block6_out').output
    feature = Conv2D(256, kernel_size=1, use_bias=False)(p3)
    feature = BatchNormalization()(feature)
    model = Model(inputs=inputs, outputs=[p0, p1, p2, feature])
    return model


# https://keras.io/zh/layers/writing-your-own-keras-layers/
class DepthwiseConv2D(Layer):

    def call(self, inputs):
        x, kernel = inputs

        n, xh, xw, c = x.shape
        x = tf.transpose(x, (1, 2, 0, 3))
        x = K.reshape(x, (1, xh, xw, -1))

        n, kh, kw, c = kernel.shape

        kernel = tf.transpose(kernel, (1, 2, 0, 3))
        kernel = K.reshape(kernel, (kh, kw, -1, 1))

        # https://www.tensorflow.org/api_docs/python/tf/nn/depthwise_conv2d
        out = K.depthwise_conv2d(x, kernel)

        _, oh, ow, _ = out.shape
        out = K.reshape(out, (oh, ow, -1, c))
        out = tf.transpose(out, (2, 0, 1, 3))
        return out

    def compute_output_shape(self, input_shape):
        print('xxx:', input_shape)
        return None, None, None, 256


def _depth_corr(kernel, search):
    kernel = Conv2D(256, kernel_size=3, use_bias=False)(kernel)
    kernel = BatchNormalization()(kernel)
    kernel = Activation('relu')(kernel)

    search = Conv2D(256, kernel_size=3, use_bias=False)(search)
    search = BatchNormalization()(search)
    search = Activation('relu')(search)

    feature = DepthwiseConv2D()([search, kernel])
    return feature


def depth_corr(kernel, search, out_channels):
    corr_feature = _depth_corr(kernel, search)
    feature = Conv2D(256, kernel_size=1, use_bias=False, name='conv2d')(corr_feature)
    feature = BatchNormalization(name='batchnormalization')(feature)
    feature = Activation('relu')(feature)
    feature = Conv2D(out_channels, kernel_size=1)(feature)
    return corr_feature, feature


def up(template, search):
    return depth_corr(template, search, out_channels=1)[1]


def mask_corr(template, search):
    return depth_corr(template, search, 63 ** 2)


class Reshape(Layer):

    def __init__(self, shape, **config):
        self.shape = list(shape)
        super().__init__(**config)

    def call(self, inputs):
        return K.reshape(inputs, [-1] + self.shape)

    # https://stackoverflow.com/questions/58678836/notimplementederror-layers-with-arguments-in-init-must-override-get-conf
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'shape': self.shape
        })
        return config


def refine(features, corr_feature):
    p0, p1, p2 = features

    # https://www.tensorflow.org/api_docs/python/tf/image/extract_patches
    p0 = tf.image.extract_patches(p0, sizes=(1, 61, 61, 1), strides=[1, 4, 4, 1], rates=[1, 1, 1, 1], padding='VALID')
    p1 = tf.image.extract_patches(p1, sizes=(1, 31, 31, 1), strides=[1, 2, 2, 1], rates=[1, 1, 1, 1], padding='VALID')
    p2 = tf.image.extract_patches(p2, sizes=(1, 15, 15, 1), strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='VALID')
    p0 = Reshape((61, 61, 64))(p0)
    p1 = Reshape((31, 31, 256))(p1)
    p2 = Reshape((15, 15, 512))(p2)

    p3 = Reshape((1, 1, 256))(corr_feature)
    out = Conv2DTranspose(32, 15, strides=15)(p3)
    h2 = Conv2D(32, 3, padding='same', activation='relu')(out)
    h2 = Conv2D(32, 3, padding='same', activation='relu')(h2)
    p2 = Conv2D(128, 3, padding='same', activation='relu')(p2)
    p2 = Conv2D(32, 3, padding='same', activation='relu')(p2)
    out = Add()([h2, p2])
    # https://www.tensorflow.org/api_docs/python/tf/image/resize
    out = tf.image.resize(out, [31, 31])
    out = Conv2D(16, 3, padding='same')(out)

    h1 = Conv2D(16, 3, padding='same', activation='relu')(out)
    h1 = Conv2D(16, 3, padding='same', activation='relu')(h1)
    p1 = Conv2D(64, 3, padding='same', activation='relu')(p1)
    p1 = Conv2D(16, 3, padding='same', activation='relu')(p1)
    out = Add()([h1, p1])
    out = tf.image.resize(out, [61, 61])
    out = Conv2D(4, 3, padding='same')(out)

    h0 = Conv2D(4, 3, padding='same', activation='relu')(out)
    h0 = Conv2D(4, 3, padding='same', activation='relu')(h0)
    p0 = Conv2D(16, 3, padding='same', activation='relu')(p0)
    p0 = Conv2D(4, 3, padding='same', activation='relu')(p0)
    out = Add()([h0, p0])
    out = tf.image.resize(out, [127, 127])
    out = Conv2D(1, 3, padding='same')(out)

    return out


def select_mask_logistic_loss(true, pred):
    pred = K.reshape(pred, (-1, 17, 17, 127 * 127))
    true = K.reshape(true, (-1, 17, 17, 127 * 127))
    # https://www.tensorflow.org/api_docs/python/tf/math/softplus
    loss = tf.math.softplus(-pred * true)
    return loss


def build_model():
    template = Input(shape=(127, 127, 3))
    search = Input(shape=(255, 255, 3))
    features = res_down()
    template_feature = features(template)[3]
    p0, p1, p2, search_feature = features(search)
    corr_feature = _depth_corr(template_feature, search_feature)
    # scores = up(template_feature, search_feature)
    # corr_feature, feature = mask_corr(template_feature, search_feature)
    masks = refine((p0, p1, p2), corr_feature)
    model = Model(inputs=[template, search], outputs=masks)
    return model


class Dataset:

    def __init__(self, fn='train.zip'):
        self.zf = zipfile.ZipFile(fn)
        with open('meta.json') as fp:
            self.meta = json.load(fp)

    def preprocess_inputs(self, mask):
        mask = (mask == self.colors).all(axis=2)
        mask.dtype = 'uint8'
        return mask

    @staticmethod
    def preprocess_mask(mask):
        masks = np.zeros((17 * 17, 127, 127), dtype='float32')
        for i in range(17 * 17):
            x = i % 17
            y = i // 17
            m = mask[y * 8:y * 8 + 127, x * 8:x * 8 + 127]
            masks[i] = (m - 0.5) * 2
        return masks

    def _generator(self, is_train):
        n = int(0.9 * len(self.meta))
        # print('nnnnnn:', n)
        if is_train:
            meta = self.meta[:n]
            random.shuffle(meta)
        else:
            meta = self.meta[n:]
        for corps in meta:
            path = corps['path']
            color = corps['color']

            if len(corps['fns']) < 2:
                continue
            frame = random.choice(corps['fns'])
            fn = f'train/Annotations/{path}/{frame}.png'
            with self.zf.open(fn) as fp:
                im = utils.imdecode(fp, 0)
            bbox = utils.find_bbox((im == color).astype('uint8'))
            fn = f'train/JPEGImages/{path}/{frame}.jpg'
            with self.zf.open(fn) as fp:
                im = utils.imdecode(fp)

            if is_train:
                mv = (np.random.random(2) - 0.5) * 2 * 4
            else:
                mv = 0, 0
            border = im.mean(axis=(0, 1))
            template, _ = utils.get_object(im, bbox, 127, move=mv, border=border)

            if is_train and random.random() < 0.12:
                grayed = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                grayed.shape = grayed.shape + (1,)
                template[:] = grayed

            template = template.astype('float32')

            frame = random.choice(corps['fns'])
            fn = f'train/Annotations/{path}/{frame}.png'
            with self.zf.open(fn) as fp:
                im = utils.imdecode(fp, 0)
            mask = im == color
            mask.dtype = 'uint8'
            bbox = utils.find_bbox(mask)

            if is_train:
                mv = (np.random.random(2) - 0.5) * 2 * 8
                q = 0.5 + (random.random() - 0.5) * 0.1
            else:
                mv = 0, 0
                q = 0.5
            mask = utils.get_object(mask, bbox, 255, move=mv, q=q)[0].astype('float32')
            fn = f'train/JPEGImages/{path}/{frame}.jpg'
            with self.zf.open(fn) as fp:
                im = utils.imdecode(fp)
            border = im.mean(axis=(0, 1))
            search, _ = utils.get_object(im, bbox, 255, move=mv, q=q, border=border)

            if is_train and random.random() < 0.12:
                grayed = cv2.cvtColor(search, cv2.COLOR_BGR2GRAY)
                grayed.shape = grayed.shape + (1,)
                search[:] = grayed

            search = search.astype('float32')

            template /= 255
            search /= 255
            template.shape = (1,) + template.shape
            search.shape = (1,) + search.shape
            masks = self.preprocess_mask(mask)
            masks.shape = (1,) + masks.shape

            yield (template, search), masks

    def generator(self, is_train=True):
        while True:
            yield from self._generator(is_train)


def mlearn():
    version = '1.0.0'
    dataset = Dataset()
    xy_train = dataset.generator()
    xy_test = dataset.generator(is_train=False)
    model = build_model()
    keras.utils.plot_model(model, 'model.png', show_shapes=True)
    model.summary()
    reduce_lr = ReduceLROnPlateau(verbose=1)
    mcp = ModelCheckpoint(filepath='weights.{epoch:03d}.h5')
    model.compile(optimizer=RMSprop(lr=0.0001),
                  loss=select_mask_logistic_loss)
    model.fit(xy_train,
              steps_per_epoch=5814,
              epochs=100,
              validation_data=xy_test,
              validation_steps=646,
              callbacks=[reduce_lr, mcp])
    model.save(f'weights.{version}.h5', include_optimizer=False)
    result = model.evaluate_generator(xy_test, steps=500)
    print(result)


def main(template, search):
    template = utils.preprocess_input(template)
    search = utils.preprocess_input(search)
    model = keras.models.load_model('weights.077.h5',
        {'DepthwiseConv2D': DepthwiseConv2D, 'Reshape': Reshape}, compile=False)
    masks = model.predict([template, search])
    np.savez('result.npz', masks=masks)


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


if __name__ == '__main__':
    if len(sys.argv) > 2:
        template = cv2.imread(sys.argv[1])
        search = cv2.imread(sys.argv[2])
        main(template, search)
    else:
        mlearn()
