from tensorflow.keras import layers, Model
import numpy as np
from Classificaltion_Evaluation import net_evaluation


def EADSPP_YoloV9(input_shape=(32, 32, 3), num_classes=3,sol=None):
    act = ['linear', 'relu', 'leaky relu', 'tanh', 'sigmoid', 'softmax']
    inputs = layers.Input(shape=input_shape)

    # Convolutional Backbone
    x = layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same')(inputs)
    x = layers.LeakyReLU(negative_slope=0.1)(x)

    x = layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)

    x = layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)

    # EADSPP Attention module
    pool1 = layers.GlobalAveragePooling2D()(x)
    pool2 = layers.GlobalMaxPooling2D()(x)
    attention = layers.concatenate([pool1, pool2])
    attention = layers.Dense(int(sol[0]), activation=act[int(sol[2])])(attention)
    attention = layers.Dense(128, activation='sigmoid')(attention)
    attention = layers.Reshape((1, 1, 128))(attention)

    x = layers.Multiply()([x, attention])

    # Detection head with upsampling to match input resolution
    x = layers.Conv2D(num_classes, (1, 1), activation='sigmoid', padding='same')(x)
    output = layers.UpSampling2D(size=(8, 8), interpolation='bilinear')(x)  # (4x upsample height and width)

    model = Model(inputs=inputs, outputs=output)
    return model


def Model_EADSPP_YoloV9(x, y, sol=None):
    if sol is None:
        sol = [5, 5, 1]

    input_shape = (32, 32, 3)
    num_classes = 3

    # Resize all datasets to 32x32x3 (if not already)
    def resize_dataset(dataset):
        resized = np.zeros((dataset.shape[0], *input_shape))
        for i in range(dataset.shape[0]):
            resized[i] = np.resize(dataset[i], input_shape)
        return resized

    x = resize_dataset(x)
    y = resize_dataset(y)
    # Create and compile model
    model = EADSPP_YoloV9(input_shape=input_shape, num_classes=num_classes, sol=sol)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # Train model
    model.fit(x, y, batch_size=8, epochs=int(sol[1]), validation_split=0.25)
    pred = model.predict(x, verbose=2)
    pred = np.asarray(pred)
    Eval = [net_evaluation(y[n].astype('uint8'), pred[n].astype('uint8')) for n in range(pred.shape[0])]
    EVAl = np.mean(Eval, axis=0)
    return EVAl, pred



