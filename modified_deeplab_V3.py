# -*- coding:utf-8 -*-
# https://github.com/rishizek/tensorflow-deeplab-v3/blob/master/deeplab_model.py
# https://kuklife.tistory.com/121
# https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_utils.py
# https://github.com/google-research/tf-slim/blob/master/tf_slim/layers/layers.py
# Random crop??
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import AveragePooling2D, Lambda, Conv2D, Conv2DTranspose, Activation, Reshape, concatenate, Concatenate, BatchNormalization, ZeroPadding2D
from resnet.resnet50 import ResNet50


def Upsample(tensor, size):
    '''bilinear upsampling'''
    name = tensor.name.split('/')[0] + '_upsample'

    def bilinear_upsample(x, size):
        resized = tf.image.resize(
            images=x, size=size)
        return resized
    y = Lambda(lambda x: bilinear_upsample(x, size),
               output_shape=size, name=name)(tensor)
    return y


def ASPP(tensor, name):
    '''atrous spatial pyramid pooling'''
    dims = K.int_shape(tensor)

    y_pool = AveragePooling2D(pool_size=(
        dims[1], dims[2]), name='average_pooling' + name)(tensor)
    y_pool = Conv2D(filters=256, kernel_size=1, padding='same',
                    kernel_initializer='he_normal', name='pool_1x1conv2d'+ name, use_bias=False)(y_pool)
    y_pool = BatchNormalization(name='bn_1'+ name)(y_pool)
    y_pool = Activation('relu', name='relu_1'+ name)(y_pool)

    y_pool = Upsample(tensor=y_pool, size=[dims[1], dims[2]])

    y_1 = Conv2D(filters=256, kernel_size=1, dilation_rate=1, padding='same',
                 kernel_initializer='he_normal', name='ASPP_conv2d_d1'+ name, use_bias=False)(tensor)
    y_1 = BatchNormalization(name='bn_2'+ name)(y_1)
    y_1 = Activation('relu', name='relu_2'+ name)(y_1)

    y_6 = Conv2D(filters=256, kernel_size=3, dilation_rate=6, padding='same',
                 kernel_initializer='he_normal', name='ASPP_conv2d_d6'+ name, use_bias=False)(tensor)
    y_6 = BatchNormalization(name='bn_3'+ name)(y_6)
    y_6 = Activation('relu', name='relu_3'+ name)(y_6)

    y_12 = Conv2D(filters=256, kernel_size=3, dilation_rate=12, padding='same',
                  kernel_initializer='he_normal', name='ASPP_conv2d_d12'+ name, use_bias=False)(tensor)
    y_12 = BatchNormalization(name='bn_4'+ name)(y_12)
    y_12 = Activation('relu', name='relu_4'+ name)(y_12)

    y_18 = Conv2D(filters=256, kernel_size=3, dilation_rate=18, padding='same',
                  kernel_initializer='he_normal', name='ASPP_conv2d_d18'+ name, use_bias=False)(tensor)
    y_18 = BatchNormalization(name='bn_5'+ name)(y_18)
    y_18 = Activation('relu', name='relu_5'+ name)(y_18)

    y = concatenate([y_pool, y_1, y_6, y_12, y_18], name='ASPP_concat'+ name)

    y = Conv2D(filters=256, kernel_size=1, dilation_rate=1, padding='same',
               kernel_initializer='he_normal', name='ASPP_conv2d_final'+ name, use_bias=False)(y)
    y = BatchNormalization(name='bn_final'+ name)(y)
    y = Activation('relu', name='relu_final'+ name)(y)
    return y


def DeepLabV3Plus(img_height, img_width, nclasses=66):
    print('*** Building DeepLabv3Plus Network ***')

    base_model = ResNet50(input_shape=(
        img_height, img_width, 3), weights='imagenet', include_top=False)

    image_features = base_model.get_layer('activation_39').output
    x_a = ASPP(image_features, "_1")
    x_a = Upsample(tensor=x_a, size=[img_height // 4, img_width // 4])

    x_b = base_model.get_layer('activation_9').output
    x_b = Conv2D(filters=48, kernel_size=1, padding='same',
                 kernel_initializer='he_normal', name='low_level_projection', use_bias=False)(x_b)
    x_b = BatchNormalization(name='bn_low_level_projection')(x_b)
    x_b = Activation('relu', name='low_level_activation')(x_b)

    x = concatenate([x_a, x_b], name='decoder_concat')

    x = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu',
               kernel_initializer='he_normal', name='decoder_conv2d_1', use_bias=False)(x)
    x = BatchNormalization(name='bn_decoder_1')(x)
    x = Activation('relu', name='activation_decoder_1')(x)

    x = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu',
               kernel_initializer='he_normal', name='decoder_conv2d_2', use_bias=False)(x)
    x = BatchNormalization(name='bn_decoder_2')(x)
    x = Activation('relu', name='activation_decoder_2')(x)
    x = Upsample(x, [img_height, img_width])

    x = Conv2D(nclasses, (1, 1), name='output_layer')(x)
    '''
    x = Activation('softmax')(x) 
    tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    Args:
        from_logits: Whether `y_pred` is expected to be a logits tensor. By default,
        we assume that `y_pred` encodes a probability distribution.
    '''     
    model = Model(inputs=base_model.input, outputs=x, name='DeepLabV3_Plus')
    model.load_weights("C:/Users/Yuhwan/Downloads/last_epoch.h5")



    image_features2 = base_model.get_layer("activation_39").output
    x_a2 = ASPP(image_features2, "_2")
    x_a2 = Upsample(tensor=x_a2, size=[img_height // 4, img_width // 4])

    x_b2 = base_model.get_layer('activation_9').output
    x_b2 = Conv2D(filters=48, kernel_size=1, padding='same',
                 kernel_initializer='he_normal', name='low_level_projection2', use_bias=False)(x_b2)
    x_b2 = BatchNormalization(name='bn_low_level_projection2')(x_b2)
    x_b2 = Activation('relu', name='low_level_activation2')(x_b2)

    x2 = concatenate([x_a2, x_b2], name='decoder_concat2')

    x2 = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu',
               kernel_initializer='he_normal', name='decoder_conv2d_3', use_bias=False)(x2)
    x2 = BatchNormalization(name='bn_decoder_3')(x2)
    x2 = Activation('relu', name='activation_decoder_3')(x2)

    x2 = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu',
               kernel_initializer='he_normal', name='decoder_conv2d_4', use_bias=False)(x2)
    x2 = BatchNormalization(name='bn_decoder_4')(x2)
    x2 = Activation('relu', name='activation_decoder_4')(x2)
    x2 = Upsample(x2, [img_height, img_width])

    x2 = Conv2D(nclasses, (1, 1), name='output_layer2')(x2)
    model2 = Model(inputs=base_model.input, outputs=x2, name='DeepLabV3_Plus2')
    model2.load_weights("C:/Users/Yuhwan/Downloads/last_epoch.h5")

    final_model = Model(inputs=base_model.input, outputs=[model.outputs, model2.outputs])

    print('*** Output_Shape => {model.output_shape} ***')
    return final_model

model = DeepLabV3Plus(512, 512, 34)
model.summary()

class LearningRateScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self,
                 init_lr, 
                 warmup_step, 
                 decay_fn):
        super(LearningRateScheduler, self).__init__()
        self.init_lr = init_lr
        self.warmup_step = warmup_step
        self.decay_fn = decay_fn
        self.current_learning_rate = tf.Variable(initial_value=init_lr, trainable=False, dtype=tf.float32)

    def __call__(self, step):
        if step == 0:
            step += 1
            
        step_float = tf.cast(step, tf.float32)
        warmup_step_float = tf.cast(self.warmup_step, tf.float32)

        self.current_learning_rate.assign(tf.cond(
            step_float < warmup_step_float,
            lambda: self.init_lr * (step_float / warmup_step_float),
            lambda: self.decay_fn(step_float - warmup_step_float),            
            ))

        return self.current_learning_rate


def step_decay(lr=3e-4, max_epochs=100, warmup=False):
    """
    step decay.
    :param lr: initial lr
    :param max_epochs: max epochs
    :param warmup: warm up or not
    :return: current lr
    """
    drop = 0.1
    max_epochs = max_epochs - 5 if warmup else max_epochs

    def decay(epoch):
        lrate = lr * np.power(drop, np.floor((1 + epoch) / max_epochs))
        return lrate

    return decay


def poly_decay(lr=3e-4, max_epochs=100, warmup=False):
    """
    poly decay.
    :param lr: initial lr
    :param max_epochs: max epochs
    :param warmup: warm up or not
    :return: current lr
    """
    max_epochs = max_epochs - 5 if warmup else max_epochs

    def decay(epoch):
        lrate = lr * (1 - np.power(epoch / max_epochs, 0.9))
        return lrate

    return decay


def cosine_decay(max_epochs, max_lr, min_lr=1e-7, warmup=False):
    """
    cosine annealing scheduler.
    :param max_epochs: max epochs
    :param max_lr: max lr
    :param min_lr: min lr
    :param warmup: warm up or not
    :return: current lr
    """
    max_epochs = max_epochs - 5 if warmup else max_epochs

    def decay(epoch):
        lrate = min_lr + (max_lr - min_lr) * (
                1 + np.cos(np.pi * epoch / max_epochs)) / 2
        return lrate

    return decay
