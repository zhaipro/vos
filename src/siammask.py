import json
import os
import sys
import zipfile
import random

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
    # inputs = Input(shape=(255, 255, 3))
    inputs = Input(shape=(None, None, 3))
    rn50 = keras.applications.ResNet50(include_top=False, input_tensor=inputs)
    # rn50.summary()
    # exit()
    '''
    for layer in rn50.layers:           # 需要训练吗？
        layer.trainable = False
    '''
    feature = rn50.layers[-95].output  # 只需要前三层, -33?
    feature = Conv2D(256, kernel_size=1, use_bias=False)(feature)
    feature = BatchNormalization()(feature)
    model = Model(inputs=inputs, outputs=feature)
    return model


# https://keras.io/zh/layers/writing-your-own-keras-layers/
class DepthwiseConv2D(Layer):

    def call(self, inputs):
        x, kernel = inputs

        # kernel = kernel[:, 4:-4, 4:-4]

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

    true = K.reshape(true, (-1, 255, 255, 1))
    print('e', pred.shape, true.shape)
    true = tf.compat.v1.extract_image_patches(true, ksizes=(1, 127, 127, 1), strides=[1, 8, 8, 1], rates=[1, 1, 1, 1], padding='VALID')
    print('b', pred.shape, true.shape)
    weight = K.mean(true, axis=-1)
    print('weight.shape:', weight.shape)
    true = (true * 2) - 1
    loss = K.log(1 + K.exp(-pred * true))
    loss = K.mean(loss, axis=-1)
    loss = loss * weight
    loss = K.sum(loss) / K.sum(weight)
    # loss = K.log(1 + K.exp(-pred[:, 8, 8] * true[:, 8, 8]))
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
        with open('meta.json') as fp:
            self.meta = json.load(fp)

    def preprocess_inputs(self, mask):
        mask = (mask == self.colors).all(axis=2)
        mask.dtype = 'uint8'
        return mask

    def _generator(self):
        for corps in self.meta:
            path = corps['path']
            color = corps['color']
            fake = random.random() < 0.3
            fake = False

            frame = random.choice(corps['fns'])
            fn = f'train/Annotations/{path}/{frame}.png'
            with self.zf.open(fn) as fp:
                im = utils.imdecode(fp, 0)
            bbox = utils.find_bbox((im == color).astype('uint8'))
            fn = f'train/JPEGImages/{path}/{frame}.jpg'
            with self.zf.open(fn) as fp:
                im = utils.imdecode(fp)
            template = utils.get_object(im, bbox, 127).astype('float32')

            if fake:
                idx = random.randint(0, len(self.meta) - 1)
                corps = self.meta[idx]
                path = corps['path']
                color = corps['color']

            frame = random.choice(corps['fns'])
            fn = f'train/Annotations/{path}/{frame}.png'
            with self.zf.open(fn) as fp:
                im = utils.imdecode(fp, 0)
            mask = im == color
            mask.dtype = 'uint8'
            bbox = utils.find_bbox(mask)
            mask = utils.get_object(mask, bbox, 255).astype('float32')
            fn = f'train/JPEGImages/{path}/{frame}.jpg'
            with self.zf.open(fn) as fp:
                im = utils.imdecode(fp)
            search = utils.get_object(im, bbox, 255).astype('float32')

            if fake:
                mask[:] = 0

            # cv2.imwrite('search.jpg', search)
            # cv2.imwrite('mask.jpg', mask * 255)
            # cv2.imwrite('template.jpg', template)
            # exit()

            template /= 255
            search /= 255
            template.shape = (1,) + template.shape
            search.shape = (1,) + search.shape
            mask.shape = (1,) + mask.shape + (1,)

            yield (template, search), mask

    def generator(self):
        while True:
            yield from self._generator()

    def demo(self):
        for (t, s), m in self.generator():
            cv2.imwrite('t.jpg', (t[0] * 255).astype('uint8'))
            cv2.imwrite('s.jpg', (s[0] * 255).astype('uint8'))
            cv2.imwrite('m.jpg', (m[0] * 255).astype('uint8'))
            exit()


def mlearn():
    version = '1.0.0'
    dataset = Dataset()
    # dataset.demo()
    xy_train = dataset.generator()
    xy_test = dataset.generator()
    model = build_model()
    model.summary()
    mcp = ModelCheckpoint(filepath='weights.{epoch:03d}.h5')
    model.compile(optimizer='rmsprop',
                  loss=select_mask_logistic_loss)
    model.fit_generator(xy_train,
                        steps_per_epoch=2000,
                        epochs=40,
                        validation_data=xy_test,
                        validation_steps=100,
                        callbacks=[mcp])
    model.save(f'weights.{version}.h5', include_optimizer=False)
    result = model.evaluate_generator(xy_test, steps=500)
    print(result)
    predict(version)


def main(template, search):
    print(template.shape, search.shape)
    template = utils.preprocess_input(template)
    search = utils.preprocess_input(search)
    print(template.shape, search.shape)
    model = keras.models.load_model('weights.012.h5', {'DepthwiseConv2D': DepthwiseConv2D}, compile=False)
    masks = model.predict([template, search])
    np.save('masks.npy', masks)


if __name__ == '__main__':
    if len(sys.argv) > 2:
        template = cv2.imread(sys.argv[1])
        search = cv2.imread(sys.argv[2])
        main(template, search)
    else:
        mlearn()
