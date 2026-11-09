import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Dense, Dropout
import tensorflow as tf
from Classificaltion_Evaluation import evaluate_Image


class ViTBlock(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_dim, ff_dim, **kwargs):
        super(ViTBlock, self).__init__(**kwargs)

        # Multi-Head Attention Layer
        self.attn = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)

        # Feed-forward network with output dimension matching the attention output (key_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation='relu'),
            Dense(key_dim)  # Output dimension should match attention output
        ])

        # Layer Normalization
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)

        # Dropout Layer for Regularization
        self.dropout = Dropout(0.1)
        self.match_channels = None

    def build(self, input_shape):
        if input_shape[-1] != self.attn.key_dim:
            self.match_channels = layers.Conv2D(self.attn.key_dim, kernel_size=1, strides=1, padding='same')

    def call(self, inputs, training=False):
        attn_output = self.attn(inputs, inputs)
        if self.match_channels is not None:
            inputs = self.match_channels(inputs)
        out1 = self.norm1(inputs + attn_output)
        ff_output = self.ffn(out1)

        ff_output = self.dropout(ff_output, training=training)

        return self.norm2(out1 + ff_output)



def encoder_with_vit(x, num_filters, kernel_size, num_heads=4, key_dim=64, ff_dim=128):
    x = ViTBlock(x, num_filters[0], kernel_size, name='layer1')
    encoder_output = [x]
    for i in range(1, len(num_filters)):
        vit_block = ViTBlock(num_heads=num_heads, key_dim=key_dim, ff_dim=ff_dim)
        layer = 'encoder_layer' + str(i)
        x = res_block(x, [num_filters[i]], kernel_size, strides=[2, 1], name=layer)
        x = vit_block(x)  # Add Vision Transformer block
        encoder_output.append(x)

    return encoder_output


def res_block(x, num_filters, kernel_size, strides, name):
    if len(num_filters) == 1:
        num_filters = [num_filters[0], num_filters[0]]

    x1 = layers.BatchNormalization()(x)
    x1 = layers.Activation('relu')(x1)
    x1 = layers.Conv2D(filters=num_filters[0], kernel_size=kernel_size, strides=strides[0], padding='same',
                       name=name + '_1')(x1)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Activation('relu')(x1)
    x1 = layers.Conv2D(filters=num_filters[1], kernel_size=kernel_size, strides=strides[1], padding='same',
                       name=name + '_2')(x1)

    x = layers.Conv2D(filters=num_filters[-1], kernel_size=1, strides=strides[0], padding='same',
                      name=name + '_shortcut')(x)
    x = layers.BatchNormalization()(x)

    x1 = layers.Add()([x, x1])

    return x1


def upsample(x, target_size):
    x_resized = layers.Lambda(lambda x: tf.image.resize(x, target_size))(x)
    return x_resized


def decoder_with_attention(x, encoder_output, num_filters, kernel_size):
    for i in range(1, len(num_filters) + 1):
        layer = 'decoder_layer' + str(i)
        target_size = encoder_output[-i].shape[1:3]
        x = upsample(x, target_size)
        x = layers.Concatenate(axis=-1)([x, encoder_output[-i]])
        x = res_block(x, [num_filters[-i]], kernel_size, strides=[1, 1], name=layer)

    return x


def vit_resunet(input_size, num_filters, kernel_size, num_channels, num_classes, num_heads=4, key_dim=64, ff_dim=128):
    x = layers.Input(shape=[input_size, input_size, num_channels])

    encoder_output = encoder_with_vit(x, num_filters, kernel_size, num_heads=num_heads, key_dim=key_dim, ff_dim=ff_dim)
    bridge = res_block(encoder_output[-1], [num_filters[-1] * 2], kernel_size, strides=[2, 1], name='bridge')
    decoder_output = decoder_with_attention(bridge, encoder_output, num_filters, kernel_size)
    output = layers.Conv2D(num_classes, kernel_size, strides=1, padding='same', name='output')(decoder_output)
    model = tf.keras.Model(x, output)
    return model


def Model_Spatial_ViT_ResUnet(Images, GT, HN=None, sol=None):
    if sol is None:
        sol = [4, 50, 0, 5, 0]
    if HN is None:
        HN = 64
    IMG_SIZE = 224
    num_filters = [64, 128, 256, 512]
    kernel_size = 3
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

    model = vit_resunet(input_size=IMG_SIZE, num_filters=num_filters, kernel_size=kernel_size, num_channels=3,
                        num_classes=1)
    model.compile(optimizer=optimizer[int(sol[2])], loss=keras.losses.binary_crossentropy, metrics=['accuracy'])

    model.summary()
    model.fit(Train_X, Train_Y, epochs=sol[1], steps_per_epoch=2, verbose="auto")
    Predict = model.predict(Train_X)

    Eval = [evaluate_Image(Train_Y[n].astype('uint8'), Predict[n].astype('uint8')) for n in range(Predict.shape[0])]
    EVAl = np.mean(Eval, axis=0)
    return Predict, EVAl


