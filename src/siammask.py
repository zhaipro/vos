# http://www.robots.ox.ac.uk/~luca/siamese-fc.html
import json
import os
import sys
import zipfile
import random

import cv2
import numpy as np
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import BatchNormalization, MaxPooling2D, concatenate
from tensorflow.keras.layers import Conv2D, Input, Activation, UpSampling2D, Dropout
# from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.losses import binary_crossentropy, mean_squared_error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.optimizers import RMSprop
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

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

        # kernel = kernel[:, 4:-4, 4:-4]

        n, xh, xw, c = x.shape
        x = tf.transpose(x, (1, 2, 0, 3))
        x = K.reshape(x, (1, xh, xw, -1))

        n, kh, kw, c = kernel.shape
        # kernel = K.reshape(kernel, (-1, kh * kw * c))
        # kernel = K.softmax(kernel)
        # print('softmax.kernel.shape:', kernel.shape)
        # kernel = K.reshape(kernel, (-1, kh, kw, c))

        print('kernel.shape:', kernel.shape)
        kernel = tf.transpose(kernel, (1, 2, 0, 3))
        kernel = K.reshape(kernel, (kh, kw, -1, 1))

        print('kernel.shape:', kernel.shape)
        print('x.shape:', x.shape)
        # https://www.tensorflow.org/api_docs/python/tf/nn/depthwise_conv2d
        out = K.depthwise_conv2d(x, kernel)

        print('out.shape:', out.shape)

        _, oh, ow, _ = out.shape
        out = K.reshape(out, (oh, ow, -1, c))
        out = tf.transpose(out, (2, 0, 1, 3))
        return out

    def compute_output_shape(self, input_shape):
        print('xxx:', input_shape)
        return None, None, None, 256


def _depth_corr(kernel, search):
    kernel = Conv2D(256, kernel_size=3, use_bias=False)(kernel)
    # kernel = kernel / (tf.norm(kernel) + 1e-8)
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
    feature = Conv2D(out_channels, kernel_size=1, activation='sigmoid')(feature)
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
        print('self.shape:', self.shape, type(self.shape))
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

    print(p0.shape, p1.shape, p2.shape, corr_feature.shape)
    # https://www.tensorflow.org/api_docs/python/tf/image/extract_patches
    p0 = tf.image.extract_patches(p0, sizes=(1, 61, 61, 1), strides=[1, 4, 4, 1], rates=[1, 1, 1, 1], padding='VALID')
    p1 = tf.image.extract_patches(p1, sizes=(1, 31, 31, 1), strides=[1, 2, 2, 1], rates=[1, 1, 1, 1], padding='VALID')
    p2 = tf.image.extract_patches(p2, sizes=(1, 15, 15, 1), strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='VALID')
    print(p0.shape, p1.shape, p2.shape, corr_feature.shape)
    p0 = Reshape((61, 61, 64))(p0)
    p1 = Reshape((31, 31, 256))(p1)
    p2 = Reshape((15, 15, 512))(p2)

    print(p0.shape, p1.shape, p2.shape, corr_feature.shape)
    p3 = Reshape((1, 1, 256))(corr_feature)
    print('pp3.shape', p3.shape)
    # p3 = K.sum(p3, axis=-1, keepdims=True)
    # print('op3.shape', p3.shape)
    # p3 = K.sigmoid(p3)
    out = Conv2DTranspose(32, 15, strides=15)(p3)
    h2 = Conv2D(32, 3, padding='same', activation='relu')(out)
    h2 = Conv2D(32, 3, padding='same', activation='relu')(h2)
    p2 = Conv2D(128, 3, padding='same', activation='relu')(p2)
    p2 = Conv2D(32, 3, padding='same', activation='relu')(p2)
    print('hahaha:', h2.shape, p2.shape)
    out = Add()([h2, p2])
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
    print('hahaha:', h0.shape, p0.shape)
    out = Add()([h0, p0])
    out = tf.image.resize(out, [127, 127])
    out = Conv2D(1, 3, padding='same', activation='sigmoid')(out)

    # print('out and p3', out.shape, p3.shape)
    # out = out * p3
    # out = Mul(out, p3)

    print('out.shape:', out.shape)

    return out


def select_mask_logistic_loss(true, pred):
    print('c', pred.shape, true.shape)
    # pred = K.reshape(pred, (-1, 127, 127, 1))
    print('d', pred.shape, true.shape)
    # https://www.tensorflow.org/api_docs/python/tf/image/resize
    # pred = tf.image.resize(pred, [127, 127])
    print('a', pred.shape, true.shape)
    pred = K.reshape(pred, (-1, 17, 17, 127 * 127))
    true = K.reshape(true, (-1, 17, 17, 127 * 127))
    # pred = pred[:, 5:-5, 5:-5]
    # true = true[:, 5:-5, 5:-5]
    print('e', pred.shape, true.shape)
    # true = tf.image.extract_patches(true, sizes=(1, 127, 127, 1), strides=[1, 8, 8, 1], rates=[1, 1, 1, 1], padding='VALID')
    print('b', pred.shape, true.shape)
    # weight = K.mean(true, axis=-1)
    # print('weight.shape:', weight.shape)
    # true = (true * 2) - 1
    # pred = K.tanh(pred)
    # soft_margin_loss
    # https://www.tensorflow.org/api_docs/python/tf/math/softplus
    loss = tf.math.softplus(-pred * true)
    # pred = K.sigmoid(pred)
    # loss = binary_crossentropy(true, pred)
    # loss = K.mean(loss, axis=-1)
    # loss = K.mean(loss * weight)
    # loss = K.sum(loss) / K.sum(weight)
    # loss = K.log(1 + K.exp(-pred[:, 8, 8] * true[:, 8, 8]))
    print('loss:', loss.shape)
    return loss


def select_score_logistic_loss(true, pred):
    print('score_c', pred.shape, true.shape)
    pred = K.reshape(pred, (-1, 17, 17, 1))
    print('score_d', pred.shape, true.shape)

    true = K.reshape(true, (-1, 17, 17, 1))
    print('score_e', pred.shape, true.shape)

    loss = binary_crossentropy(true, pred)
    # loss = tf.math.softplus(-pred * true)
    print('score_loss:', loss.shape)
    return loss


def select_mask_logistic_loss_v1(true, pred):
    print('c', pred.shape, true.shape)
    pred = K.reshape(pred, (-1, 63, 63, 1))
    print('d', pred.shape, true.shape)
    pred = tf.image.resize(pred, [127, 127])
    print('a', pred.shape, true.shape)
    pred = K.reshape(pred, (-1, 17, 17, 127 * 127))

    true = K.reshape(true, (-1, 255, 255, 1))
    print('e', pred.shape, true.shape)
    true = tf.image.extract_patches(true, sizes=(1, 127, 127, 1), strides=[1, 8, 8, 1], rates=[1, 1, 1, 1], padding='VALID')
    print('b', pred.shape, true.shape)
    # weight = K.mean(true, axis=-1)
    # print('weight.shape:', weight.shape)
    # true = (true * 2) - 1
    # pred = K.tanh(pred)
    pred = K.sigmoid(pred)
    loss = binary_crossentropy(true, pred)
    # loss = K.mean(loss, axis=-1)
    # loss = K.mean(loss * weight)
    # loss = K.sum(loss) / K.sum(weight)
    # loss = K.log(1 + K.exp(-pred[:, 8, 8] * true[:, 8, 8]))
    print('loss:', loss.shape)
    return loss


def build_model():
    template = Input(shape=(127, 127, 3))
    search = Input(shape=(255, 255, 3))
    features = res_down()
    template_feature = features(template)[3]
    p0, p1, p2, search_feature = features(search)
    corr_feature = _depth_corr(template_feature, search_feature)
    scores = up(template_feature, search_feature)
    # corr_feature, feature = mask_corr(template_feature, search_feature)
    masks = refine((p0, p1, p2), corr_feature)
    # model = Model(inputs=[template, search], outputs=masks)
    model = Model(inputs=[template, search], outputs=[masks, scores])
    return model


class Dataset:

    def __init__(self, fn='train.zip'):
        self.zf = zipfile.ZipFile(fn)
        with open('meta.json') as fp:
            self.meta = json.load(fp)

    @staticmethod
    def get_weight(mask):
        w = mask.sum()
        weight = np.zeros(17 * 17, dtype='float32')
        for i in range(17 * 17):
            x = i % 17
            y = i // 17
            a = mask[:, y * 8:y * 8 + 127, x * 8:x * 8 + 127].sum() / w
            weight[i] = a < 0.01 or a > 0.96
        weight.shape = 1, 17, 17, 1
        return weight

    @staticmethod
    def get_weights():
        weights = np.zeros((17, 17, 127, 127), dtype='float32')
        weight = np.zeros((255, 255), dtype='float32')
        for i in range(17 * 17):
            x = i % 17
            y = i // 17
            weight[y * 8:y * 8 + 127, x * 8:x * 8 + 127] += 1
        for i in range(17 * 17):
            x = i % 17
            y = i // 17
            127 * 127 * 17 * 17
            weights[y, x] = weight[y * 8:y * 8 + 127, x * 8:x * 8 + 127] / (17 * 17)
        return weight * 255 / weight.max(), weights

    @staticmethod
    def _preprocess_mask(mask):
        masks = np.zeros((17 * 17, 127, 127), dtype='float32')
        for i in range(17 * 17):
            x = i % 17
            y = i // 17
            m = mask[y * 8:y * 8 + 127, x * 8:x * 8 + 127]
            # masks[i] = (m - 0.5) * 2
            masks[i] = m
        return masks

    @staticmethod
    def preprocess_mask(mask):
        # h, w = mask.shape
        ms, ss = 0, 0
        mask_weight = np.zeros(17 * 17, dtype='float32')
        score_weight = np.zeros(17 * 17, dtype='float32')
        weight = mask.sum()
        masks = np.zeros((17 * 17, 127, 127), dtype='float32')
        scores = np.zeros(17 * 17)
        if weight == 0:
            scores[:] = 0
            score_weight[:] = 1
            mask_weight.shape = 1, 17, 17, 1
            score_weight.shape = 1, 17, 17, 1
            return masks, scores, (mask_weight, score_weight)
        for i in range(17 * 17):
            x = i % 17
            y = i // 17
            m = mask[y * 8:y * 8 + 127, x * 8:x * 8 + 127]
            score = m.sum() / weight
            # masks[i] = (m - 0.5) * 2
            # masks[i] = m
            if score > 0.96:
                masks[i] = m
                scores[i] = 1
                score_weight[i] = 1
                mask_weight[i] = 1
                ss += 1
                ms += 1
                # print(a, weight, outputs[i].min(), outputs[i].max())
            elif score < 0.69:
                score_weight[i] = 1
                # masks[i] = -1
                # scores[i] = 0
                ss += 1
            # else:
                # masks[i] = m - 1
                # scores[i] = 0
        mask_weight.shape = 1, 17, 17, 1
        score_weight.shape = 1, 17, 17, 1
        # mask_weight *= 17 * 17 / (ms + 1e-8)
        # score_weight *= 17 * 17 / (ss + 1e-8)
        return masks, scores, (mask_weight, score_weight)

    def __generator(self, is_train):
        n = int(0.9 * len(self.meta))
        if is_train:
            meta = self.meta[:n]
            random.shuffle(meta)
        else:
            meta = self.meta[n:]
        for corps in meta:
            path = corps['path']
            color = corps['color']
            fake = random.random() < 0.3
            # fake = False
            flip = False
            # if is_train:
            #     flip = random.random() < 0.1
            # else:
            #     flip = False

            if len(corps['fns']) < 2:
                continue
            # _frame = random.randint(0, len(corps['fns']) - 2)
            # frame = corps['fns'][_frame]
            frame = random.choice(corps['fns'])
            fn = f'train/Annotations/{path}/{frame}.png'
            with self.zf.open(fn) as fp:
                im = utils.imdecode(fp, 0)
            mask = (im == color).astype('uint8')
            bbox = utils.find_bbox(mask)
            fn = f'train/JPEGImages/{path}/{frame}.jpg'
            with self.zf.open(fn) as fp:
                im = utils.imdecode(fp)

            if is_train:
                mv = (np.random.random(2) - 0.5) * 2 * 4
            else:
                mv = 0, 0
            mv = 0, 0
            border = im.mean(axis=(0, 1))
            template, _ = utils.get_object(im, bbox, 127, move=mv, flip=flip, border=border)
            # x1, y1, x2, y2 = bbox
            # mask[y1:y2, x1:x2] = 1
            # template_mask, _ = utils.get_object(mask, bbox, 127, move=mv, flip=flip)
            # cv2.imwrite('_template.jpg', template)
            # cv2.imwrite('_template_mask.jpg', template_mask * 255)
            # exit()

            if is_train and random.random() < 0.12:
                grayed = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                grayed.shape = grayed.shape + (1,)
                template[:] = grayed

            if fake:
                idx = random.randint(0, len(self.meta) - 1)
                corps = self.meta[idx]
                path = corps['path']
                color = corps['color']

            frame = random.choice(corps['fns'])
            # if is_train:
            #     frame = random.choice(corps['fns'])
            # else:
            #     frame = corps['fns'][_frame + 1]
            fn = f'train/Annotations/{path}/{frame}.png'
            with self.zf.open(fn) as fp:
                im = utils.imdecode(fp, 0)
            _im = im
            mask = im == color
            mask.dtype = 'uint8'
            _mask = mask
            bbox = utils.find_bbox(mask)

            if is_train:
                mv = (np.random.random(2) - 0.5) * 2 * 8   # 8 or 64
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

            if fake:
                mask[:] = 0

            template = utils.preprocess_input(template)
            search = utils.preprocess_input(search)

            try:
                masks, scores, weights = self.preprocess_mask(mask)
            except:
                np.savez('errors.npz', im=_im, mask=_mask)
                exit()
            masks.shape = (1,) + masks.shape
            scores.shape = (1,) + scores.shape

            # template_ex = np.zeros((1, 127, 127, 4), dtype='float32')
            # template_ex[..., :3] = template
            # template_ex[..., 3] = template_mask
            # template_ex[..., 3] -= 0.5
            # template_ex[..., 3] *= 2.0
            # search_ex = np.zeros((1, 255, 255, 4), dtype='float32')
            # search_ex[..., :3] = search
            # yield (template_ex, search_ex), masks

            yield (template, search), (masks, scores), weights
            # yield (template, search), masks
            # yield (template, search), mask, self.get_weight(mask)

    def _generator(self, is_train):
        while True:
            yield from self.__generator(is_train)

    def generator(self, is_train=True, batch_size=7):
        templates, searches, masks, scores, masks_weight, scores_weight = [], [], [], [], [], []
        for (template, search), (mask, score), (mask_weight, score_weight) in self._generator(is_train):
            templates.append(template)
            searches.append(search)
            masks.append(mask)
            scores.append(score)
            masks_weight.append(mask_weight)
            scores_weight.append(score_weight)
            if len(templates) == batch_size:
                yield (np.concatenate(templates), np.concatenate(searches)), (np.concatenate(masks), np.concatenate(scores)), (np.concatenate(masks_weight), np.concatenate(scores_weight))
                templates, searches, masks, scores, masks_weight, scores_weight = [], [], [], [], [], []

    def demo(self):
        for (t, s), m in self.generator():
            cv2.imwrite('t.jpg', (t[0] * 255).astype('uint8'))
            cv2.imwrite('s.jpg', (s[0] * 255).astype('uint8'))
            cv2.imwrite('m.jpg', (m[0] * 255).astype('uint8'))
            exit()


def dice_coeff(y_true, y_pred):
    # y_pred = K.reshape(y_pred, (-1, 17, 17, 127 * 127))
    # y_true = K.reshape(y_true, (-1, 17, 17, 127 * 127))

    smooth = 1.
    intersection = K.sum(y_true * y_pred)
    score = (2. * intersection + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth)
    return score


def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss


def bce_dice_loss(y_true, y_pred):
    y_pred = K.reshape(y_pred, (-1, 17, 17, 127 * 127))
    y_true = K.reshape(y_true, (-1, 17, 17, 127 * 127))

    loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss


def psnr(hr, sr, max_val=1):
    mse = K.mean(K.square(hr - sr))
    return 10.0 / np.log(10) * K.log(max_val ** 2 / mse)


def mlearn():
    version = '1.0.1'
    dataset = Dataset()
    # dataset.demo()
    xy_train = dataset.generator()
    xy_test = dataset.generator(is_train=False)

    model = build_model()
    # with tf.device('/cpu:0'):
    #     model = build_model()
    # model_parallel = multi_gpu_model(model, gpus=2)

    # model = keras.models.load_model('weights.001.h5',
    #     {'DepthwiseConv2D': DepthwiseConv2D, 'Reshape': Reshape}, compile=False)
    keras.utils.plot_model(model, 'model.png', show_shapes=True)
    # model_parallel.summary()
    model.summary()
    reduce_lr = ReduceLROnPlateau(verbose=1)
    mcp = ModelCheckpoint(filepath='weights.{epoch:03d}.h5')
    model.compile(optimizer=RMSprop(lr=0.0001),
                  # loss=bce_dice_loss)
                  sample_weight_mode='temporal',
                  loss=[bce_dice_loss, select_score_logistic_loss])
    model.fit_generator(xy_train,
                        steps_per_epoch=5814 // (7 * 1),
                        epochs=100,
                        validation_data=xy_test,
                        validation_steps=646 // (7 * 1),
                        callbacks=[reduce_lr, mcp])
    model.save(f'weights.{version}.h5', include_optimizer=False)
    result = model.evaluate_generator(xy_test, steps=500)
    print(result)


def main(template, search):
    print(template.shape, search.shape)
    template = utils.preprocess_input(template)
    search = utils.preprocess_input(search)
    print(template.shape, search.shape)
    model = keras.models.load_model('weights.054.h5',
        {'DepthwiseConv2D': DepthwiseConv2D, 'Reshape': Reshape}, compile=False)
    masks, scores = model.predict([template, search])
    print(masks.shape)
    np.savez('result.npz', masks=masks, scores=scores)


def depreprocess(masks):
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


if __name__ == '__main__':
    # weight, weights = Dataset.get_weights()
    # print(weight.shape, weight.dtype, weight.max())
    # cv2.imshow('a', weight.astype('uint8'))
    # cv2.waitKey()
    # exit()
    # os.makedirs('results', exist_ok=True)
    # model = keras.models.load_model('weights.081.h5',
    #     {'DepthwiseConv2D': DepthwiseConv2D, 'Reshape': Reshape}, compile=False)
    # i = 0
    # for (template, search), masks_true in Dataset().generator(is_train=False, batch_size=1):
    #     cv2.imwrite(f'results/{i}_template.jpg', template[0, ..., :3] * 255)
    #     cv2.imwrite(f'results/{i}_search.jpg', search[0, ..., :3] * 255)
    #     masks_prev = model.predict([template, search])
    #     masks_true = depreprocess(masks_true)
    #     masks_prev = depreprocess(masks_prev)
    #     cv2.imwrite(f'results/{i}_mask_true.jpg', masks_true * 255)
    #     cv2.imwrite(f'results/{i}_mask_prev.jpg', masks_prev * 255)
    #     i += 1
    #     if i >= 646:
    #         exit()
    # model = build_model()
    # model.summary()
    # exit()
    if len(sys.argv) > 2:
        template = cv2.imread(sys.argv[1])
        search = cv2.imread(sys.argv[2])
        main(template, search)
    else:
        mlearn()
