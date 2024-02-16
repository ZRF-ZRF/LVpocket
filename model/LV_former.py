import numpy as np
import h5py
import math
from openbabel import pybel
import openbabel

import tfbio.net
import tfbio.data
from tfbio.data import Featurizer

from skimage.segmentation import clear_border
from skimage.measure import label
from skimage.morphology import closing
import keras

from keras.layers import Conv3D, Conv3DTranspose, PReLU, Convolution3D, BatchNormalization
from keras.layers import Input, Softmax, Dropout, Add, Lambda, Layer, Reshape, Concatenate
from keras.models import Model
from keras import backend as K
from keras.regularizers import l2
import tensorflow as tf
from .data import DataWrapper, get_box_size


__all__ = [
    'dice',
    'dice_np',
    'dice_loss',
    'ovl',
    'ovl_np',
    'ovl_loss',
    'LV_former'

]


def dice(y_true, y_pred, smoothing_factor=0.01):
    """Dice coefficient adapted for continuous data (predictions) computed with
    keras layers.
    """

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)

    return ((2. * intersection + smoothing_factor)
            / (K.sum(y_true_f) + K.sum(y_pred_f) + smoothing_factor))


def dice_np(y_true, y_pred, smoothing_factor=0.01):
    """Dice coefficient adapted for continuous data (predictions) computed with
    numpy arrays.
    """

    intersection = (y_true * y_pred).sum()
    total = y_true.sum() + y_pred.sum()

    return ((2. * intersection + smoothing_factor)
            / (total + smoothing_factor))


def dice_loss(y_true, y_pred):
    """Keras loss function for Dice coefficient (loss(t, y) = -dice(t, y))"""
    return 1. - dice(y_true, y_pred)



def ovl(y_true, y_pred, smoothing_factor=0.01):
    """Overlap coefficient computed with keras layers"""
    concat = K.concatenate((y_true, y_pred))
    return ((K.sum(K.min(concat, axis=-1)) + smoothing_factor)
            / (K.sum(K.max(concat, axis=-1)) + smoothing_factor))


def ovl_np(y_true, y_pred, smoothing_factor=0.01):
    """Overlap coefficient computed with numpy arrays"""
    concat = np.concatenate((y_true, y_pred), axis=-1)
    return ((concat.min(axis=-1).sum() + smoothing_factor)
            / (concat.max(axis=-1).sum() + smoothing_factor))


def ovl_loss(y_true, y_pred):
    """Keras loss function for overlap coefficient (loss(t, y) = -ovl(t, y))"""
    return -ovl(y_true, y_pred)

def gelu(tensor):
    g = Lambda(lambda x: tf.pow(x, 3))(tensor)
    g = Lambda(lambda x: tf.multiply(x, 0.044715))(g)
    g = Add()([tensor, g])
    g = Lambda(lambda x: tf.multiply(x, 0.7978845608028654))(g)
    g = Lambda(lambda x: tf.tanh(x))(g)
    g = Lambda(lambda x: tf.add(x, 1))(g)
    g = Lambda(lambda x: tf.multiply(x[0], x[1]))([tensor, g])
    g = Lambda(lambda x: tf.multiply(x, 0.5))(g)
    return g

def attention_layer(query, key, value, d_model, heads, dropout=0.1):
    shape = value.shape.as_list()
    size = shape[1]
    depth = (d_model // heads)

    q = Conv3D(filters=d_model, kernel_size=3, padding='same', kernel_regularizer=l2(1e-3))(query)
    k = Conv3D(filters=d_model, kernel_size=3, padding='same', kernel_regularizer=l2(1e-3))(key)
    v = Conv3D(filters=d_model, kernel_size=3, padding='same', kernel_regularizer=l2(1e-3))(value)

    q = Reshape((heads, size * size * size, depth))(q)
    k = Reshape((heads, size * size * size, depth))(k)
    v = Reshape((heads, size * size * size, depth))(v)

    logits = Lambda(lambda x: tf.matmul(x[0], x[1], transpose_b=True))([q, k])

    logits = Lambda(lambda x: tf.multiply(x, 1.0 / math.sqrt(float(depth))))(logits)

    weights = Softmax()(logits)
    weights = Dropout(dropout)(weights)
    attention_outpout = Lambda(lambda x: tf.matmul(x[0], x[1]))([weights, v])
    attention_outpout = Reshape((size, size, size, d_model))(attention_outpout)
    attention_outpout = Conv3D(filters=d_model, kernel_size=3, padding='same',
                               kernel_regularizer=l2(1e-3))(attention_outpout)
    attention_outpout = Add()([query, attention_outpout])

    return attention_outpout


class LayerNorm(Layer):
    def __init__(self,
                 center=True,
                 scale=False,
                 epsilon=None,
                 gamma_initializer='ones',
                 beta_initializer='zeros',
                 gamma_regularizer=None,
                 beta_regularizer=None,
                 gamma_constraint=None,
                 beta_constraint=None,
                 **kwargs,
                 ):
        super(LayerNorm, self).__init__(**kwargs)
        self.supports_masking = True
        self.center = center
        self.scale = scale
        if epsilon is None:
            epsilon = K.epsilon() * K.epsilon()
        self.epsilon = epsilon
        self.gamma_initializer = keras.initializers.get(gamma_initializer)
        self.beta_initializer = keras.initializers.get(beta_initializer)
        self.gamma_regularizer = keras.regularizers.get(gamma_regularizer)
        self.beta_regularizer = keras.regularizers.get(beta_regularizer)
        self.gamma_constraint = keras.constraints.get(gamma_constraint)
        self.beta_constraint = keras.constraints.get(beta_constraint)
        self.gamma, self.beta = 0., 0.

    def call(self, inputs, **kwargs):
        mean = K.mean(inputs, axis=-1, keepdims=True)
        variance = K.mean(K.square(inputs - mean), axis=-1, keepdims=True)
        std = K.sqrt(variance + self.epsilon)
        outputs = (inputs - mean) / std
        if self.scale:
            outputs *= self.gamma
        if self.center:
            outputs += self.beta
        return outputs


def feedforward(tensor, d_model, d_ff, dropout=0.1):
    x1 = Conv3D(filters=d_ff, kernel_size=3, padding='same')(tensor)
    x1 = gelu(x1)
    x1 = Dropout(dropout)(x1)
    x1 = BatchNormalization()(x1)
    x1 = Conv3D(filters=d_model, kernel_size=3, padding='same')(x1)
    x1 = Dropout(dropout)(x1)

    x = Add()([tensor, x1])
    x = BatchNormalization()(x)
    return x


class PositionEncoding(Layer):
    def __init__(self,  d_model=0, **kwargs):
        self.d_model = d_model
        super(PositionEncoding, self).__init__(**kwargs)

    def call(self, inputs):
        seq_length = inputs.shape[1]
        model_dim = int(inputs.shape[-1])
        print(model_dim)
        position_encodings = np.zeros((seq_length, seq_length, seq_length, model_dim))
        for pos1 in range(seq_length):
            for pos2 in range(seq_length):
                for pos3 in range(seq_length):
                    for i in range(model_dim):
                        position_encodings[pos1, pos2, pos3, i] = pos3 / np.power(10000, (i - i % 2) / model_dim)
        position_encodings[:, :, :, 0::2] = np.sin(position_encodings[:, :, :, 0::2])
        position_encodings[:, :, :, 1::2] = np.cos(position_encodings[:, :, :, 1::2])
        position_encodings = K.cast(position_encodings, 'float32')
        position_encodings = position_encodings + inputs
        return position_encodings

class EncoderLayer(Layer):
    def __init__(self, d_model, d_ff, head, dropout=0.1, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)
        self.d_ff = d_ff
        self.d_model = d_model
        self.head = head
        self.dropout = dropout

    def __call__(self, tensor, *args, **kwargs):
        tensor = Conv3D(filters=self.d_model, kernel_size=1, padding='same', kernel_regularizer=l2(1e-3))(tensor)

        tensor = PositionEncoding(self.d_model)(tensor)

        attn_output = attention_layer(query=tensor, key=tensor, value=tensor, d_model=self.d_model, heads=self.head,
                                      dropout=self.dropout)
        attn_output = Dropout(self.dropout)(attn_output)
        attn_output = Add()([tensor, attn_output])
        out1 = LayerNorm()(attn_output)

        ffn_output = feedforward(out1, d_model=self.d_model, d_ff=self.d_ff, dropout=self.dropout)
        ffn_output = Dropout(self.dropout)(ffn_output)
        ffn_output = Add()([out1, ffn_output])
        out2 = LayerNorm()(ffn_output)

        return Add()([tensor, out2])

class LV_NET(Model):

    def identity_block(self, input_tensor, filters, stage, block, layer=None):
        filter1, filter2, filter3 = filters
        if K.image_data_format() == 'channels_last':  # （样本数，行数（高），列数（宽），通道数）
            bn_axis = 4
        else:
            bn_axis = 1  # channel_first:（样本数，通道数，行数（高），列数（宽））
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        x = Convolution3D(filters=filter1, kernel_size=1, name=conv_name_base + '2a', kernel_regularizer=l2(1e-4))(
            input_tensor)
        print({'identity_x1': x})
        if layer == None:
            x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
            print({'bn1': x})
        #x = Activation('relu')(x)
        x = PReLU()(x)

        x = Convolution3D(filters=filter2, kernel_size=3, padding='same', name=conv_name_base + '2b',
                          kernel_regularizer=l2(1e-4))(x)
        print({'identity_x2': x})
        if layer == None:
            x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
            #print({'bn2': x})
        x = PReLU()(x)

        x = Convolution3D(filters=filter3, kernel_size=1, name=conv_name_base + '2c', kernel_regularizer=l2(1e-4))(x)
        print({'identity_x3': x})
        if layer == None:
            x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
            print({'bn3': x})
        x = Add()([x, input_tensor])
        print({'add': x})
        x = PReLU()(x)

        return x


    DEFAULT_SIZE = 36
    def __init__(self, inputs=None, outputs=None, data_handle=None,
                 featurizer=None, scale=None, box_size=None, input_channels=None,
                 output_channels=None, l2_lambda=1e-3, **kwargs):
        if data_handle is not None:
            if not isinstance(data_handle, DataWrapper):
                raise TypeError('data_handle should be a DataWrapper object,'
                                ' got %s instead' % type(data_handle))
            # data_handle必须为DataWrapper类型

            if box_size is None:
                box_size = data_handle.box_size
            elif box_size != data_handle.box_size:
                raise ValueError('specified box_size does not match '
                                 'data_handle.box_size (%s != %s)'
                                 % (box_size, data_handle.box_size))

            if input_channels is None:
                input_channels = data_handle.x_channels
            elif input_channels != data_handle.x_channels:
                raise ValueError('specified input_channels does not match '
                                 'data_handle.x_channels (%s != %s)'
                                 % (input_channels, data_handle.x_channels))

            if output_channels is None:
                output_channels = data_handle.y_channels
            elif output_channels != data_handle.y_channels:
                raise ValueError('specified output_channels does not match '
                                 'data_handle.y_channels (%s != %s)'
                                 % (output_channels, data_handle.y_channels))
            if scale is None:
                self.scale = data_handle.scale
            elif scale != data_handle.scale:
                raise ValueError('specified scale does not match '
                                 'data_handle.scale (%s != %s)'
                                 % (scale, data_handle.scale))
            self.max_dist = data_handle.max_dist
        else:
            self.scale = scale
            self.max_dist = None    # we'll calculate it later from box size

        if featurizer is not None:
            if not isinstance(featurizer, tfbio.data.Featurizer):
                raise TypeError('featurizer should be a tfbio.data.Featurizer '
                                'object, got %s instead' % type(featurizer))
            if input_channels is None:
                input_channels = len(featurizer.FEATURE_NAMES)
            elif input_channels != len(featurizer.FEATURE_NAMES):
                raise ValueError(
                    'specified input_channels or data_handle.x_channels does '
                    'not match number of features produce by featurizer '
                    '(%s != %s)' % (input_channels, len(featurizer.FEATURE_NAMES)))

        if inputs is not None:
            if outputs is None:
                raise ValueError('you must provide both inputs and outputs')
            if isinstance(inputs, list):
                i_shape = LV_former.__total_shape(inputs)
            else:
                i_shape = inputs.shape

            if isinstance(outputs, list):
                o_shape = LV_former.__total_shape(outputs)
            else:
                o_shape = outputs.shape

            if len(i_shape) != 5:
                raise ValueError('input should be 5D, got %sD instead'
                                 % len(i_shape))
            elif len(o_shape) != 5:
                raise ValueError('output should be 5D, got %sD instead'
                                 % len(o_shape))
            elif i_shape[1:4] != o_shape[1:4]:
                raise ValueError('input and output shapes do not match '
                                 '(%s != %s)' % (i_shape[1:4], o_shape[1:4]))
            if box_size is None:
                box_size = i_shape[1]
            elif i_shape[1:4] != (box_size,) * 3:
                raise ValueError('input shape does not match box_size '
                                 '(%s != %s)' % (i_shape[1:4], (box_size,) * 3))

            if input_channels is not None and i_shape[4] != input_channels:
                raise ValueError('number of channels (specified via featurizer'
                                 ', input_channels or data_handle) does not '
                                 'match input shape (%s != %s)'
                                 % (i_shape[4], input_channels))
            if output_channels is not None and o_shape[4] != output_channels:
                raise ValueError('specified output_channels or '
                                 'data_handle.y_channels does not match '
                                 'output shape (%s != %s)'
                                 % (o_shape[4], output_channels))
        else:
            if outputs is not None:
                raise ValueError('you must provide both inputs and outputs')
            elif (box_size is None or input_channels is None
                  or output_channels is None):
                raise ValueError('you must either provide: 1) inputs and '
                                 'outputs (keras layers); 2) data_handle '
                                 '(DataWrapper object); 3) box_size, '
                                 'input_channels and output_channels')
            elif (box_size < self.DEFAULT_SIZE
                  or box_size % self.DEFAULT_SIZE != 0):
                raise ValueError('box_size does not match the default '
                                 'architecture. Pleas scecify inputs and outputs')

            inputs = Input((box_size, box_size, box_size, input_channels), name='lvnet_input')

            conv1 = Conv3D(18, 5, padding='same', name='conv1')(inputs)
            conv1 = PReLU(name='conv1_prelu')(conv1)

            concat1 = Concatenate(name='concat1')([inputs, conv1])
            down_conv1 = Conv3D(36, [2, 2, 1], strides=2, name='down_conv_1')(concat1)
            down_conv1 = PReLU(name='down_conv1_prelu')(down_conv1)

            conv2_1 = Conv3D(36, 5, padding='same', name='conv2_1')(down_conv1)
            conv2_1 = PReLU(name='conv2_1_Pelu')(conv2_1)
            conv2_2 = Conv3D(36, 5, padding='same', name='conv2_2')(conv2_1)
            conv2_2 = PReLU(name='conv2_2_Prelu')(conv2_2)

            concat2 = Concatenate(name='concat2')([down_conv1, conv2_2])

            down_conv2 = Conv3D(72, [2, 2, 1], strides=2, name='down_conv2')(concat2)
            down_conv2 = PReLU(name='down_conv2_Prelu')(down_conv2)

            conv3_1 = Conv3D(72, 5, padding='same', name='conv3_1')(down_conv2)
            conv3_1 = PReLU(name='conv3_1_relu')(conv3_1)
            conv3_2 = Conv3D(72, 5, padding='same', name='conv3_2')(conv3_1)
            conv3_2 = PReLU(name='conv3_2_relu')(conv3_2)
            conv3_3 = Conv3D(72, 5,padding='same',name='conv3_3')(conv3_2)
            conv3_3 = PReLU(name='conv3_3_relu')(conv3_3)   # connect attention

            concat3 = Concatenate(name='concat3')([conv3_3, down_conv2])

            down_conv3 = Conv3D(144,[2, 2, 2], strides=2,name='down_conv3')(concat3)
            down_conv3 = PReLU(name='down_conv3_relu')(down_conv3)

            conv4_1 = Conv3D(144,5,padding='same',name='conv4_1')(down_conv3)
            conv4_1 = PReLU(name='conv4_1_relu')(conv4_1)
            conv4_2 = Conv3D(144, 5, padding='same', name='conv4_2')(conv4_1)
            conv4_2 = PReLU(name='conv4_2_relu')(conv4_2)
            conv4_3 = Conv3D(144, 5, padding='same', name='conv4_3')(conv4_2)
            conv4_3 = PReLU(name='conv4_3_relu')(conv4_3)

            concat4 = Concatenate(name='concat4')([down_conv3, conv4_3])

            down_conv4 = Conv3D(288, [2, 2, 1], strides=2, name='down_conv4')(concat4)
            down_conv4 = PReLU(name='down_conv4_relu')(down_conv4)

            " The 'L' pathway"
            LV_1 = Conv3D(18, 5, padding='same', name='LV-conv1')(inputs)
            LV_prelu_1 = PReLU(name='LV_prelu1')(LV_1)

            LV_concat1 = Concatenate(name='LV-concat1')([inputs, LV_prelu_1])

            LV_down_1 = Conv3D(36, [2, 2 , 1], strides=2, name='LV-down1')(LV_concat1)
            LV_prelu_2 = PReLU(name='LV-prelu2')(LV_down_1)

            LV_2 = Conv3D(36, 5, padding='same', name='LV-conv2')(LV_prelu_2)
            LV_prelu_3 = PReLU(name='LV-prelu3')(LV_2)

            LV_concat2 = Concatenate(name='LV-concat2')([LV_prelu_2, LV_prelu_3])

            LV_down_2 = Conv3D(72, [2, 2, 1], strides=2, name='LV-down2')(LV_concat2)
            LV_prelu_4 = PReLU(name='LV-prelu4')(LV_down_2)

            LV_3 = Conv3D(72, 5, padding='same', name='LV-conv3')(LV_prelu_4)
            LV_prelu_5 = PReLU(name='LV-prelu5')(LV_3)

            LV_concat3 = Concatenate(name='LV-concat3')([LV_prelu_4, LV_prelu_5])

            LV_atten3 = EncoderLayer(d_model=72, d_ff=144, head=8, dropout=0.1)(LV_concat3)

            LV_atten4 = EncoderLayer(d_model=72, d_ff=144, head=8, dropout=0.1)(conv3_3)

            atten_concat1 = Concatenate(name='atten-concat1')([LV_atten3, LV_atten4])

            LV_down_3 = Conv3D(144, [2, 2, 2], strides=2, name='LV-down3')(atten_concat1)
            LV_prelu_6 = PReLU(name='LV-prelu6')(LV_down_3)

            LV_4 = Conv3D(144, 5, padding='same', name='LV-conv4')(LV_prelu_6)
            LV_prelu_7 = PReLU(name='LV-prelu7')(LV_4)

            LV_down_4 = Conv3D(288, [2, 2, 1], strides=2, name='LV-down4')(LV_prelu_7)
            LV_prelu_8 = PReLU(name='LV-prelu8')(LV_down_4)

            LV_5 = Conv3D(288, 5, padding='same', name='LV-conv5')(LV_prelu_8)
            LV_prelu_9 = PReLU(name='LV-prelu9')(LV_5)

            conv5_1 = Conv3D(288, 5, padding='same', name='conv5_1')(down_conv4)
            conv5_1 = PReLU(name='conv5_1_relu')(conv5_1)
            conv5_2 = Conv3D(288, 5, padding='same', name='conv5_2')(conv5_1)
            conv5_2 = PReLU(name='conv5_2_relu')(conv5_2)
            conv5_3 = Conv3D(288, 5, padding='same', name='conv5_3')(conv5_2)
            conv5_3 = PReLU(name='conv5_3_relu')(conv5_3)

            concat5 = Concatenate(name='concat5')([conv5_3, LV_prelu_9])

            up_conv5 = Conv3DTranspose(288, [2, 2, 1], strides=2, name='up_conv5',kernel_regularizer=l2(l2_lambda))(concat5)
            up_conv5 = PReLU(name='up_conv5_relu')(up_conv5)

            concat4_1 = self.identity_block(concat4, [288, 288, 288], stage=1, block='c')

            conv6 = Concatenate(name='conv6')([up_conv5, concat4_1])

            conv6_1 = Conv3D(288, 5, padding='same', name='conv6_1')(conv6)
            conv6_1 = PReLU(name='conv6_1_relu')(conv6_1)
            conv6_2 = Conv3D(288, 5, padding='same', name='conv6_2')(conv6_1)
            conv6_2 = PReLU(name='conv6_2_relu')(conv6_2)
            conv6_3 = Conv3D(288, 5, padding='same', name='conv6_3')(conv6_2)
            conv6_3 = PReLU(name='conv6_3_relu')(conv6_3)

            concat6 = Concatenate(name='concat6')([up_conv5, conv6_3])

            up_conv6 = Conv3DTranspose(144, [3 , 3, 3], strides=2,name='up_conv6')(concat6)
            up_conv6 = PReLU(name='up_conv6_relu')(up_conv6)

            concat3_1 = self.identity_block(concat3, [144, 144, 144], stage=2, block='d')

            conv7 = Concatenate(name='conv7')([up_conv6, concat3_1])

            conv7_1 = Conv3D(144, 5,padding='same',name='conv7_1')(conv7)
            conv7_1 = PReLU(name=f'conv7_1_relu')(conv7_1)
            conv7_2 = Conv3D(144, 5, padding='same', name='conv7_2')(conv7_1)
            conv7_2 = PReLU(name='conv7_2_relu')(conv7_2)
            conv7_3 = Conv3D(144, 5, padding='same', name='conv7_3')(conv7_2)
            conv7_3 = PReLU(name='conv7_3_relu')(conv7_3)

            concat7 = Concatenate(name='concat7')([up_conv6, conv7_3])

            up_conv7 = Conv3DTranspose(72, [2, 2, 1], strides=2,name='up_conv7')(concat7)
            up_conv7 = PReLU(name='up_conv7_relu')(up_conv7)

            concat2_1 = self.identity_block(concat2, [72, 72, 72], stage=3, block='e')

            conv8 = Concatenate(name='conv8')([up_conv7, concat2_1])

            conv8_1 = Conv3D(72, 5, padding='same', name='conv8_1')(conv8)
            conv8_1 = PReLU(name='conv8_1_relu')(conv8_1)
            conv8_2 = Conv3D(72, 5, padding='same', name='conv8_2')(conv8_1)
            conv8_2 = PReLU(name='conv8_2_relu')(conv8_2)

            concat8 = Concatenate(name='concat8')([conv8_2, up_conv7])

            up_conv8 = Conv3DTranspose(36, [2, 2, 1], strides=2, name='up_conv8')(concat8)
            up_conv8 = PReLU(name='up_conv8_relu')(up_conv8)

            conv9 = Concatenate(name='conv9')([up_conv8, concat1])

            conv9_1 = Conv3D(36, 5, padding='same', name='conv9_1')(conv9)

            concat9 = Concatenate(name='concat9')([conv9_1, up_conv8])


            outputs = Conv3D(
                filters=1,
                kernel_size=1,
                kernel_regularizer=l2(1e-4),
                activation='sigmoid',
                name='pocket'
            )(concat9)
            print({'outputs': outputs})
        super().__init__(inputs=inputs, outputs=outputs, **kwargs)
        self.data_handle = data_handle
        self.featurizer = featurizer
        if self.max_dist is None and self.scale is not None:
            self.max_dist = (box_size - 1) / (2 * self.scale)

    @staticmethod
    def __total_shape(tensor_list):
        if len(tensor_list) == 1:
            total_shape = tuple(tensor_list[0].shape.as_list())
        else:
            total_shape = (*tensor_list[0].shape.as_list()[:-1],
                           sum(t.shape.as_list()[-1] for t in tensor_list))
        return total_shape

    def save_keras(self, path):
        class_name = self.__class__.__name__
        self.__class__.__name__ = 'Model'
        self.save(path, include_optimizer=False)
        self.__class__.__name__ = class_name

    @staticmethod
    def load_model(path, **attrs):
        """Load model saved in HDF format"""
        from keras.models import load_model as keras_load
        custom_objects = {name: val for name, val in globals().items()
                          if name in __all__}
        custom_objects['LayerNorm'] = LayerNorm
        custom_objects['PositionEncoding'] = PositionEncoding
        custom_objects['tf'] = tf
        custom_objects['math'] = math
        model = keras_load(path, custom_objects=custom_objects)
        #model = keras_load(path)

        if 'data_handle' in attrs:
            if not isinstance(attrs['data_handle'], DataWrapper):
                raise TypeError('data_handle should be a DataWrapper object, '
                                'got %s instead' % type(attrs['data_handle']))
            elif 'scale' not in attrs:
                attrs['scale'] = attrs['data_handle'].scale
            elif attrs['scale'] != attrs['data_handle'].scale:
                raise ValueError('specified scale does not match '
                                 'data_handle.scale (%s != %s)'
                                 % (attrs['scale'], attrs['data_handle'].scale))

            if 'featurizer' in attrs:
                if not (isinstance(attrs['featurizer'], tfbio.data.Featurizer)):
                    raise TypeError(
                        'featurizer should be a tfbio.data.Featurizer object, '
                        'got %s instead' % type(attrs['featurizer']))
                elif (len(attrs['featurizer'].FEATURE_NAMES)
                      != attrs['data_handle'].x_channels):
                    raise ValueError(
                        'number of features produced be the featurizer does '
                        'not match data_handle.x_channels (%s != %s)'
                        % (len(attrs['featurizer'].FEATURE_NAMES),
                           attrs['data_handle'].x_channels))

            if 'max_dist' not in attrs:
                attrs['max_dist'] = attrs['data_handle'].max_dist
            elif attrs['max_dist'] != attrs['data_handle'].max_dist:
                raise ValueError('specified max_dist does not match '
                                 'data_handle.max_dist (%s != %s)'
                                 % (attrs['max_dist'],
                                    attrs['data_handle'].max_dist))

            if 'box_size' not in attrs:
                attrs['box_size'] = attrs['data_handle'].box_size
            elif attrs['box_size'] != attrs['data_handle'].box_size:
                raise ValueError('specified box_size does not match '
                                 'data_handle.box_size (%s != %s)'
                                 % (attrs['box_size'],
                                    attrs['data_handle'].box_size))

        elif 'featurizer' in attrs and not (isinstance(attrs['featurizer'],
                                                       tfbio.data.Featurizer)):
            raise TypeError(
                'featurizer should be a tfbio.data.Featurizer object, '
                'got %s instead' % type(attrs['featurizer']))

        if 'scale' in attrs and 'max_dist' in attrs:
            box_size = get_box_size(attrs['scale'], attrs['max_dist'])
            if 'box_size' in attrs:
                if not attrs['box_size'] == box_size:
                    raise ValueError('specified box_size does not match '
                                     'size defined by scale and max_dist (%s != %s)'
                                     % (attrs['box_size'], box_size))
            else:
                attrs['box_size'] = box_size

        # TODO: add some attrs validation if handle is not specified

        for attr, value in attrs.items():
            setattr(model, attr, value)
        return model

    def pocket_density_from_mol(self, mol):
        """Predict porobability density of pockets using pybel.Molecule object
        as input"""

        if not isinstance(mol, pybel.Molecule):
            raise TypeError('mol should be a pybel.Molecule object, got %s '
                            'instead' % type(mol))
        if self.featurizer is None:
            raise ValueError('featurizer must be set to make predistions for '
                             'molecules')
        if self.scale is None:
            raise ValueError('scale must be set to make predistions')
        prot_coords, prot_features = self.featurizer.get_features(mol)
        centroid = prot_coords.mean(axis=0)
        prot_coords -= centroid
        resolution = 1. / self.scale
        x = tfbio.data.make_grid(prot_coords, prot_features,
                                 max_dist=self.max_dist,
                                 grid_resolution=resolution)

        density = self.predict(x)

        origin = (centroid - self.max_dist)
        step = np.array([1.0 / self.scale] * 3)
        return density, origin, step

    def pocket_density_from_grid(self, pdbid):
        """Predict porobability density of pockets using 3D grid (np.ndarray)
        as input"""

        if self.data_handle is None:
            raise ValueError('data_handle must be set to make predictions '
                             'using PDBIDs')
        if self.scale is None:
            raise ValueError('scale must be set to make predistions')
        x, _ = self.data_handle.prepare_complex(pdbid)
        origin = (self.data_handle[pdbid]['centroid'][:] - self.max_dist)
        step = np.array([1.0 / self.scale] * 3)
        density = self.predict(x)
        return density, origin, step

    def save_density_as_cmap(self, density, origin, step, fname='pockets.cmap',
                             mode='w', name='protein'):
        """Save predcited pocket density as .cmap file (which can be opened in
        UCSF Chimera or ChimeraX)
        """
        if len(density) != 1:
            raise ValueError('saving more than one prediction at a time is not'
                             ' supported')
        density = density[0].transpose([3, 2, 1, 0])

        with h5py.File(fname, mode) as cmap:
            g1 = cmap.create_group('Chimera')
            for i, channel_dens in enumerate(density):
                g2 = g1.create_group('image%s' % (i + 1))
                g2.attrs['chimera_map_version'] = 1
                g2.attrs['name'] = name.encode() + b' binding sites'
                g2.attrs['origin'] = origin
                g2.attrs['step'] = step
                g2.create_dataset('data_zyx', data=channel_dens,
                                  shape=channel_dens.shape,
                                  dtype='float32')

    def save_density_as_cube(self, density, origin, step, fname='pockets.cube',
                             mode='w', name='protein'):
        """Save predcited pocket density as .cube file (format originating from
        Gaussian package).
        """
        angstrom2bohr = 1.889725989

        if len(density) != 1:
            raise ValueError('saving more than one prediction at a time is not'
                             ' supported')
        if density.shape[-1] != 1:
            raise NotImplementedError('saving multichannel density is not'
                                      ' supported yet, please save each'
                                      ' channel in a separate file.')

        with open(fname, 'w') as f:
            f.write('%s CUBE FILE.\n' % name)
            f.write('OUTER LOOP: X, MIDDLE LOOP: Y, INNER LOOP: Z\n')
            f.write('    1 %12.6f %12.6f %12.6f\n' % tuple(angstrom2bohr * origin))
            f.write(
                '%5i %12.6f     0.000000      0.000000\n'
                '%5i     0.000000 %12.6f      0.000000\n'
                '%5i     0.000000      0.000000 %12.6f\n'
                % tuple(i for pair in zip(density.shape[1:4],
                                          angstrom2bohr * step) for i in pair)
            )
            f.write('    1     0.000000 %12.6f %12.6f %12.6f\n'
                    % tuple(angstrom2bohr * origin))
            f.write('\n'.join([' '.join('%12.6f' % i for i in row)
                               for row in density.reshape((-1, 6))]))

    def get_pockets_segmentation(self, density, threshold=0.5, min_size=50):
        """Predict pockets using specified threshold on the probability density.
        Filter out pockets smaller than min_size A^3
        """

        if len(density) != 1:
            raise ValueError('segmentation of more than one pocket is not'
                             ' supported')

        voxel_size = (1 / self.scale) ** 3
        # get a general shape, without distinguishing output channels
        bw = closing((density[0] > threshold).any(axis=-1))

        # remove artifacts connected to border
        cleared = clear_border(bw)

        # label regions
        label_image, num_labels = label(cleared, return_num=True)
        for i in range(1, num_labels + 1):
            pocket_idx = (label_image == i)
            pocket_size = pocket_idx.sum() * voxel_size
            if pocket_size < min_size:
                label_image[np.where(pocket_idx)] = 0
        return label_image

    def predict_pocket_atoms(self, mol, dist_cutoff=4.5, expand_residue=True,
                             **pocket_kwargs):
        """Predict pockets for a given molecule and get AAs forming them
        (list pybel.Molecule objects).

        Parameters
        ----------
        mol: pybel.Molecule object
            Protein structure
        dist_cutoff: float, optional (default=2.0)
            Maximal distance between protein atom and predicted pocket
        expand_residue: bool, optional (default=True)
            Inlude whole residue if at least one atom is included in the pocket
        pocket_kwargs:
            Keyword argument passed to `get_pockets_segmentation` method

        Returns
        -------
        pocket_mols: list of pybel.Molecule objects
            Fragments of molecule corresponding to detected pockets.
        """

        from scipy.spatial.distance import cdist

        coords = np.array([a.coords for a in mol.atoms])
        atom2residue = np.array([a.residue.idx for a in mol.atoms])
        residue2atom = np.array([[a.idx - 1 for a in r.atoms]
                                 for r in mol.residues])

        # predcit pockets
        density, origin, step = self.pocket_density_from_mol(mol)
        pockets = self.get_pockets_segmentation(density, **pocket_kwargs)

        # find atoms close to pockets
        pocket_atoms = []
        for pocket_label in range(1, pockets.max() + 1):
            indices = np.argwhere(pockets == pocket_label).astype('float32')
            indices *= step
            indices += origin
            distance = cdist(coords, indices)
            close_atoms = np.where((distance < dist_cutoff).any(axis=1))[0]
            if len(close_atoms) == 0:
                continue
            if expand_residue:
                residue_ids = np.unique(atom2residue[close_atoms])
                close_atoms = np.concatenate(residue2atom[residue_ids])
            pocket_atoms.append([int(idx) for idx in close_atoms])

        # create molecules correcponding to atom indices
        pocket_mols = []
        # TODO optimize (copy atoms to new molecule instead of deleting?)
        for pocket in pocket_atoms:
            # copy molecule
            pocket_mol = mol.clone
            atoms_to_del = (set(range(len(pocket_mol.atoms)))
                            - set(pocket))
            pocket_mol.OBMol.BeginModify()
            for aidx in sorted(atoms_to_del, reverse=True):
                atom = pocket_mol.OBMol.GetAtom(aidx + 1)
                pocket_mol.OBMol.DeleteAtom(atom)
            pocket_mol.OBMol.EndModify()
            pocket_mols.append(pocket_mol)

        return pocket_mols
    def save_pocket_mol2(self,mol,path,format='mol2',**pocket_kwargs):
        density, origin, step = self.pocket_density_from_mol(mol)
        pockets = self.get_pockets_segmentation(density, **pocket_kwargs)
        i=0
        for pocket_label in range(1, pockets.max() + 1):
            indices = np.argwhere(pockets == pocket_label).astype('float32')
            indices *= step
            indices += origin
            mol=openbabel.OBMol()
            for idx in indices:
                a=mol.NewAtom()
                a.SetVector(float(idx[0]),float(idx[1]),float(idx[2]))
            p_mol=pybel.Molecule(mol)
            p_mol.write(format,path+'/pocket'+str(i)+'.'+format)
            i+=1






