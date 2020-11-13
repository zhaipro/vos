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
from tensorflow.keras.layers import Add
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from tensorflow import keras
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

import utils
import resnet


def res_down():
    inputs = Input(shape=(None, None, 3))
    rn50 = resnet.ResNet50(input_tensor=inputs)
    '''
    for layer in rn50.layers:           # 需要训练吗？
        layer.trainable = False
    '''
    feature = rn50.get_layer('conv4_block6_out').output
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







def refine(features, corr_feature):
    p0, p1, p2 = features
    p3 = Reshape((1, 1, 256))(corr_feature)
    out = Conv2DTranspose(32, 15, strides=15)(p3)
    h2 = Conv2d(32, 3, padding='same', activation='relu')(out)
    h2 = Conv2d(32, 3, padding='same', activation='relu')(h2)
    p2 = Conv2d(128, 3, padding='same', activation='relu')(p2)
    p2 = Conv2d(32, 3, padding='same', activation='relu')(p2)
    out = Add()([h2, p2])
    out = tf.image.resize(out, [31, 31])
    out = Conv2d(16, 3, padding='same')(out)

    h1 = Conv2d(16, 3, padding='same', activation='relu')(out)
    h1 = Conv2d(16, 3, padding='same', activation='relu')(h1)
    p1 = Conv2d(64, 3, padding='same', activation='relu')(p1)
    p1 = Conv2d(16, 3, padding='same', activation='relu')(p1)
    out = Add()([h1, p1])
    out = tf.image.resize(out, [61, 61])
    out = Conv2d(4, 3, padding='same')(out)

    h0 = Conv2d(4, 3, padding='same', activation='relu')(out)
    h0 = Conv2d(4, 3, padding='same', activation='relu')(h1)
    p0 = Conv2d(16, 3, padding='same', activation='relu')(p0)
    p0 = Conv2d(4, 3, padding='same', activation='relu')(p0)
    out = Add()([h0, p0])
    out = tf.image.resize(out, [127, 127])
    out = Conv2d(1, 3, padding='same')(out)
    out = Reshape((127 * 127))(out)

    '''
    def __init__(self):
        super(Refine, self).__init__()
        self.v0 = nn.Sequential(nn.Conv2d(64, 16, 3, padding=1), nn.ReLU(),
                           nn.Conv2d(16, 4, 3, padding=1),nn.ReLU())

        self.v1 = nn.Sequential(nn.Conv2d(256, 64, 3, padding=1), nn.ReLU(),
                           nn.Conv2d(64, 16, 3, padding=1), nn.ReLU())

        self.v2 = nn.Sequential(nn.Conv2d(512, 128, 3, padding=1), nn.ReLU(),
                           nn.Conv2d(128, 32, 3, padding=1), nn.ReLU())

        self.h2 = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
                           nn.Conv2d(32, 32, 3, padding=1), nn.ReLU())

        self.h1 = nn.Sequential(nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
                           nn.Conv2d(16, 16, 3, padding=1), nn.ReLU())

        self.h0 = nn.Sequential(nn.Conv2d(4, 4, 3, padding=1), nn.ReLU(),
                           nn.Conv2d(4, 4, 3, padding=1), nn.ReLU())

        self.deconv = nn.ConvTranspose2d(256, 32, 15, 15)

        self.post0 = nn.Conv2d(32, 16, 3, padding=1)
        self.post1 = nn.Conv2d(16, 4, 3, padding=1)
        self.post2 = nn.Conv2d(4, 1, 3, padding=1)

        for modules in [self.v0, self.v1, self.v2, self.h2, self.h1, self.h0, self.deconv, self.post0, self.post1, self.post2,]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    nn.init.kaiming_uniform_(l.weight, a=1)

    def forward(self, f, corr_feature, pos=None, test=False):
        p0 = F.unfold(f[0], (61, 61), padding=0, stride=4).permute(0, 2, 1).contiguous().view(-1, 64, 61, 61)
        if not (pos is None): p0 = torch.index_select(p0, 0, pos)
        p1 = F.unfold(f[1], (31, 31), padding=0, stride=2).permute(0, 2, 1).contiguous().view(-1, 256, 31, 31)
        if not (pos is None): p1 = torch.index_select(p1, 0, pos)
        p2 = F.unfold(f[2], (15, 15), padding=0, stride=1).permute(0, 2, 1).contiguous().view(-1, 512, 15, 15)
        if not (pos is None): p2 = torch.index_select(p2, 0, pos)

        if not(pos is None):
            p3 = corr_feature[:, :, pos[0], pos[1]].view(-1, 256, 1, 1)
        else:
            p3 = corr_feature.permute(0, 2, 3, 1).contiguous().view(-1, 256, 1, 1)

        out = self.deconv(p3)
        out = self.post0(F.upsample(self.h2(out) + self.v2(p2), size=(31, 31)))
        out = self.post1(F.upsample(self.h1(out) + self.v1(p1), size=(61, 61)))
        out = self.post2(F.upsample(self.h0(out) + self.v0(p0), size=(127, 127)))
        out = out.view(-1, 127*127)
        return out
    '''






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

    def _generator(self, is_train):
        if is_train:
            meta = self.meta[:-100]
        else:
            meta = self.meta[-100:]
        for corps in meta:
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

    def generator(self, is_train=True):
        while True:
            yield from self._generator(is_train)

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
    xy_test = dataset.generator(is_train=False)
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
