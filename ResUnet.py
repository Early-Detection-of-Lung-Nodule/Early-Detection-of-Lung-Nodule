import numpy as np
import tensorflow as tf
from tensorflow import keras
from Classificaltion_Evaluation import  evaluate_Image


def res_block_initial(x, num_filters, kernel_size, strides, name):
    if len(num_filters) == 1:
        num_filters = [num_filters[0], num_filters[0]]

    x1 = tf.keras.layers.Conv2D(filters=num_filters[0],
                                kernel_size=kernel_size,
                                strides=strides[0],
                                padding='same',
                                name=name + '_1')(x)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.Activation('relu')(x1)
    x1 = tf.keras.layers.Conv2D(filters=num_filters[1],
                                kernel_size=kernel_size,
                                strides=strides[1],
                                padding='same',
                                name=name + '_2')(x1)

    x = tf.keras.layers.Conv2D(filters=num_filters[-1],
                               kernel_size=1,
                               strides=1,
                               padding='same',
                               name=name + '_shortcut')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x1 = tf.keras.layers.Add()([x, x1])

    return x1


def res_block(x, num_filters, kernel_size, strides, name):

    if len(num_filters) == 1:
        num_filters = [num_filters[0], num_filters[0]]

    x1 = tf.keras.layers.BatchNormalization()(x)
    x1 = tf.keras.layers.Activation('relu')(x1)
    x1 = tf.keras.layers.Conv2D(filters=num_filters[0],
                                kernel_size=kernel_size,
                                strides=strides[0],
                                padding='same',
                                name=name + '_1')(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.Activation('relu')(x1)
    x1 = tf.keras.layers.Conv2D(filters=num_filters[1],
                                kernel_size=kernel_size,
                                strides=strides[1],
                                padding='same',
                                name=name + '_2')(x1)

    x = tf.keras.layers.Conv2D(filters=num_filters[-1],
                               kernel_size=1,
                               strides=strides[0],
                               padding='same',
                               name=name + '_shortcut')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x1 = tf.keras.layers.Add()([x, x1])

    return x1


def upsample(x, target_size):

    x_resized = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, target_size))(x)

    return x_resized


def encoder(x, num_filters, kernel_size):

    x = res_block_initial(x, [num_filters[0]], kernel_size, strides=[1, 1], name='layer1')

    encoder_output = [x]
    for i in range(1, len(num_filters)):
        layer = 'encoder_layer' + str(i)
        x = res_block(x, [num_filters[i]], kernel_size, strides=[2, 1], name=layer)
        encoder_output.append(x)

    return encoder_output

def decoder(x, encoder_output, num_filters, kernel_size):
    for i in range(1, len(num_filters) + 1):
        layer = 'decoder_layer' + str(i)
        target_size = encoder_output[-i].shape[1:3]
        x = upsample(x, target_size)
        print(x.shape, encoder_output[-i].shape)
        x = tf.keras.layers.Concatenate(axis=-1)([x, encoder_output[-i]])
        x = res_block(x, [num_filters[-i]], kernel_size, strides=[1, 1], name=layer)

    return x

def res_unet(input_size, num_filters, kernel_size, num_channels, num_classes):
    input_size = 224
    x = tf.keras.Input(shape=[input_size, input_size, num_channels])
    encoder_output = encoder(x, num_filters, kernel_size)
    bridge = res_block(encoder_output[-1], [num_filters[-1] * 2], kernel_size,
                       strides=[2, 1], name='bridge')

    print(encoder_output[-1].shape)
    decoder_output = decoder(bridge, encoder_output, num_filters, kernel_size)
    output = tf.keras.layers.Conv2D(num_classes,
                                    kernel_size,
                                    strides=1,
                                    padding='same',
                                    name='output')(decoder_output)

    model = tf.keras.Model(x, output)
    return model

def Model_ResUnet(Images, GT, HN=None, sol=None):
    if sol is None:
        sol = [4, 50, 0, 5, 0]
    if HN is None:
        HN = 64
    IMG_SIZE = 224
    num_filters = 64
    kernel_size = 5
    optimizer = ['SGD', 'Adam', 'RMSprop', 'Adagrad', 'Adadelta']
    input_shape = (IMG_SIZE, IMG_SIZE, 3)
    Train_Temp = np.zeros((Images.shape[0], input_shape[0], input_shape[1], input_shape[2]))
    for i in range(Images.shape[0]):
        Train_Temp[i, :] = np.resize(Images[i], (input_shape[0], input_shape[1], input_shape[2]))
    Train_X = Train_Temp.reshape(Train_Temp.shape[0], input_shape[0], input_shape[1], input_shape[2])

    Test_Temp = np.zeros((GT.shape[0], input_shape[0], input_shape[1], input_shape[2]))
    for i in range(GT.shape[0]):
        Test_Temp[i, :] = np.resize(GT[i], (input_shape[0], input_shape[1], input_shape[2]))
    Train_Y = Test_Temp.reshape(Test_Temp.shape[0], input_shape[0], input_shape[1], input_shape[2])

    encoder_output = encoder(Train_X, num_filters, kernel_size)
    model = res_block(encoder_output[-1], [num_filters[-1] * 2], kernel_size,
                       strides=[2, 1], name='bridge')
    model.compile(optimizer=optimizer[int(sol[2])], loss=keras.losses.binary_crossentropy, metrics=['accuracy'])
    model.summary()
    model.fit(Train_X, Train_Y, epochs=sol[1], steps_per_epoch=2, verbose="auto")
    Predict = model.predict(Train_X)
    Eval = [evaluate_Image(Train_Y[n].astype('uint8'), Predict[n].astype('uint8')) for n in range(Predict.shape[0])]
    EVAl = np.mean(Eval, axis=0)
    return Predict, EVAl



