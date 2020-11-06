import json
import os
import sys
import zipfile

import cv2
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.layers import BatchNormalization, MaxPooling2D, concatenate
from tensorflow.keras.layers import Conv2D, Input, Activation, UpSampling2D, Dropout
from tensorflow.keras.layers import Reshape
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from tensorflow import keras
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

import utils


def res_down():
    inputs = Input(shape=(None, None, 3))
    rn50 = keras.applications.ResNet50(include_top=False, input_tensor=inputs)
    for layer in rn50.layers:           # 需要训练吗？
        layer.trainable = False
    feature = rn50.layers[-95].output  # 只需要前三层, -33?
    feature = Conv2D(256, kernel_size=1, use_bias=False)(feature)
    feature = BatchNormalization()(feature)
    model = Model(inputs=inputs, outputs=feature)
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

        out = K.depthwise_conv2d(x, kernel)

        _, oh, ow, _ = out.shape
        out = K.reshape(out, (oh, ow, -1, c))
        out = tf.transpose(out, (2, 0, 1, 3))
        return out

    def compute_output_shape(self, input_shape):
        print('xxx:', input_shape)
        return None, None, None, 256


def depth_corr(kernel, search, out_channels):
    kernel = Conv2D(256, kernel_size=3, use_bias=False)(kernel)
    kernel = BatchNormalization()(kernel)
    kernel = Activation('relu')(kernel)

    search = Conv2D(256, kernel_size=3, use_bias=False)(search)
    search = BatchNormalization()(search)
    search = Activation('relu')(search)

    feature = DepthwiseConv2D()([search, kernel])
    feature = Conv2D(256, kernel_size=1, use_bias=False, name='conv2d')(feature)
    feature = BatchNormalization(name='batchnormalization')(feature)
    feature = Activation('relu')(feature)
    feature = Conv2D(out_channels, kernel_size=1)(feature)

    return feature


def mask_corr(template, search):
    return depth_corr(template, search, 63 ** 2)


def select_mask_logistic_loss(true, pred):
    # return pred
    print('c', pred.shape, true.shape)
    pred = K.reshape(pred, (-1, 63, 63, 1))
    print('d', pred.shape, true.shape)
    # https://www.tensorflow.org/api_docs/python/tf/image/resize
    pred = tf.image.resize(pred, [127, 127])
    print('a', pred.shape, true.shape)
    pred = K.reshape(pred, (-1, 17, 17, 127 * 127))

    # true = K.reshape(true, (-1, 255, 255, 1))
    print('e', pred.shape, true.shape)
    true = tf.compat.v1.extract_image_patches(true, ksizes=(1, 127, 127, 1), strides=[1, 8, 8, 1], rates=[1, 1, 1, 1], padding='VALID')
    print('b', pred.shape, true.shape)
    # true = tf.image.extract_patches(true, (127, 127), 8, )
    weight = K.sum(true, axis=-1, keepdims=True)
    print('weight.shape:', weight.shape)
    true = (true * 2) - 1
    loss = K.log(1 + K.exp(-pred * true)) * weight
    print('loss:', loss.shape)
    return loss


def build_model():
    template = Input(shape=(127, 127, 3))
    search = Input(shape=(255, 255, 3))
    features = res_down()
    template_feature = features(template)
    search_feature = features(search)
    outputs = mask_corr(template_feature, search_feature)
    model = Model(inputs=[template, search], outputs=outputs)
    return model


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
                    x = utils.get_object(im, bbox, 255).astype('float32')
                    y = utils.get_object(y, bbox, 127)
                    x.shape = (1,) + x.shape
                    y.shape = (1,) + y.shape + (1,)
                    # 1, 255, 255, 3   1, 127, 127, 1
                    yield [x[:, 64:-64, 64:-64].copy(), x], y.astype('float32')

    def generator(self):
        while True:
            yield from self._generator()


def mlearn():
    version = '1.0.0'
    dataset = Dataset()
    xy_train = dataset.generator()
    xy_test = dataset.generator()
    model = build_model()
    model.summary()
    mcp = ModelCheckpoint(filepath='weights.{epoch:03d}.h5')
    model.compile(optimizer='rmsprop',
                  loss=select_mask_logistic_loss)
    model.fit_generator(xy_train,
                        steps_per_epoch=29500,
                        epochs=20,
                        validation_data=xy_test,
                        validation_steps=500,
                        callbacks=[mcp])
    model.save(f'weights.{version}.h5', include_optimizer=False)
    result = model.evaluate_generator(xy_test, steps=500)
    print(result)
    predict(version)


if __name__ == '__main__':
    mlearn()
