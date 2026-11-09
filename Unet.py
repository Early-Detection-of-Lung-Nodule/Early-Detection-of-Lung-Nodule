import keras
from keras.src.models import Model
from keras.src.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout
import numpy as np

from Classificaltion_Evaluation import evaluate_Image


def unet_model(input_size=(256, 256, 3), num_classes=1):
    inputs = Input(input_size)

    # Encoder
    c1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, 3, activation='relu', padding='same')(c1)
    p1 = MaxPooling2D(pool_size=(2, 2))(c1)

    c2 = Conv2D(128, 3, activation='relu', padding='same')(p1)
    c2 = Conv2D(128, 3, activation='relu', padding='same')(c2)
    p2 = MaxPooling2D(pool_size=(2, 2))(c2)

    c3 = Conv2D(256, 3, activation='relu', padding='same')(p2)
    c3 = Conv2D(256, 3, activation='relu', padding='same')(c3)
    p3 = MaxPooling2D(pool_size=(2, 2))(c3)

    c4 = Conv2D(512, 3, activation='relu', padding='same')(p3)
    c4 = Conv2D(512, 3, activation='relu', padding='same')(c4)
    d = Dropout(0.5)(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(d)

    # Bottleneck
    c5 = Conv2D(1024, 3, activation='relu', padding='same')(p4)
    c5 = Conv2D(1024, 3, activation='relu', padding='same')(c5)
    d = Dropout(0.5)(c5)

    # Decoder
    u6 = UpSampling2D(size=(2, 2))(d)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(512, 3, activation='relu', padding='same')(u6)
    c6 = Conv2D(512, 3, activation='relu', padding='same')(c6)

    u7 = UpSampling2D(size=(2, 2))(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(256, 3, activation='relu', padding='same')(u7)
    c7 = Conv2D(256, 3, activation='relu', padding='same')(c7)

    u8 = UpSampling2D(size=(2, 2))(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(128, 3, activation='relu', padding='same')(u8)
    c8 = Conv2D(128, 3, activation='relu', padding='same')(c8)

    u9 = UpSampling2D(size=(2, 2))(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(64, 3, activation='relu', padding='same')(u9)
    c9 = Conv2D(64, 3, activation='relu', padding='same')(c9)

    # Output layer
    outputs = Conv2D(num_classes, 1, activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model


def Model_Unet(Images, GT, HN=None, sol=None):
    if sol is None:
        sol = [4, 50, 0, 5, 0]
    if HN is None:
        HN = 64
    IMG_SIZE = 256
    classes = 3
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

    model = unet_model(input_size=(256, 256, 3), num_classes=classes)
    model.compile(optimizer=optimizer[int(sol[2])], loss=keras.losses.binary_crossentropy, metrics=['accuracy'])
    model.summary()
    model.fit(Train_X, Train_Y, epochs=sol[1], steps_per_epoch=2, verbose="auto")
    Predict = model.predict(Train_X)
    Eval = [evaluate_Image(Train_Y[n].astype('uint8'), Predict[n].astype('uint8')) for n in range(Predict.shape[0])]
    EVAl = np.mean(Eval, axis=0)
    return Predict, EVAl