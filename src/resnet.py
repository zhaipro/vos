# https://github.com/keras-team/keras-applications/blob/master/keras_applications/resnet_common.py
from tensorflow.keras import backend
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import utils as keras_utils


def block1(x, filters, kernel_size=3, stride=1,
           conv_shortcut=True, dilation=1, name=None):
    """A residual block.

    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer.
        kernel_size: default 3, kernel size of the bottleneck layer.
        stride: default 1, stride of the first layer.
        conv_shortcut: default True, use convolution shortcut if True,
            otherwise identity shortcut.
        name: string, block label.

    # Returns
        Output tensor for the residual block.
    """
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    if conv_shortcut is True:
        if stride == 1:
            shortcut = layers.Conv2D(4 * filters, 1, strides=stride,
                                     name=name + '_0_conv')(x)
            shortcut = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                                 name=name + '_0_bn')(shortcut)
        else:
            shortcut = layers.Conv2D(4 * filters, 3, strides=stride,
                                     name=name + '_0_conv')(x)
            shortcut = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                                 name=name + '_0_bn')(shortcut)
    else:
        shortcut = x

    x = layers.Conv2D(filters, 1, name=name + '_1_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_1_bn')(x)
    x = layers.Activation('relu', name=name + '_1_relu')(x)

    padding = 'SAME' if stride == 1 else 'VALID'
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding=padding,
                      dilation_rate=dilation, name=name + '_2_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_2_bn')(x)
    x = layers.Activation('relu', name=name + '_2_relu')(x)

    x = layers.Conv2D(4 * filters, 1, name=name + '_3_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_3_bn')(x)

    x = layers.Add(name=name + '_add')([shortcut, x])
    x = layers.Activation('relu', name=name + '_out')(x)
    return x


def stack1(x, filters, blocks, stride1=2, dilation=1, name=None):
    """A set of stacked residual blocks.

    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer in a block.
        blocks: integer, blocks in the stacked blocks.
        stride1: default 2, stride of the first layer in the first block.
        name: string, stack label.

    # Returns
        Output tensor for the stacked blocks.
    """
    x = block1(x, filters, stride=stride1, name=name + '_block1')
    for i in range(2, blocks + 1):
        x = block1(x, filters, conv_shortcut=False, dilation=dilation,
                   name=name + '_block' + str(i))
    return x


def ResNet50(input_tensor=None, input_shape=None):
    model_name = 'resnet50'

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    x = layers.Conv2D(64, 7, strides=2, name='conv1_conv')(img_input)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name='conv1_bn')(x)
    p0 = layers.Activation('relu', name='conv1_relu')(x)
    x = layers.MaxPooling2D(3, strides=2, padding='SAME', name='pool1_pool')(p0)

    p1 = stack1(x, 64, 3, stride1=1, name='conv2')
    p2 = stack1(p1, 128, 4, stride1=2, name='conv3')
    p3 = stack1(p2, 256, 6, stride1=1, dilation=2, name='conv4')
    p4 = stack1(p3, 512, 3, stride1=1, name='conv5')
    x = p4

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = models.Model(inputs, x, name=model_name)

    return model


if __name__ == '__main__':
    a = ResNet50(input_shape=(255, 255, 3))
    a.summary()
