import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from Classificaltion_Evaluation import net_evaluation


def create_detr_model(input_shape, num_classes=1):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D(pool_size=2, padding='same')(x)
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=2, padding='same')(x)
    x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=2, padding='same')(x)

    # Upsampling layers to match the input size (32x32)
    x = layers.Conv2DTranspose(128, 3, activation='relu', strides=2, padding='same')(x)
    x = layers.Conv2DTranspose(64, 3, activation='relu', strides=2, padding='same')(x)
    output = layers.Conv2DTranspose(num_classes, 3, activation='softmax', strides=2, padding='same')(x)

    # Model definition
    model = Model(inputs=inputs, outputs=output)

    # Compile the model
    model.compile(optimizer=keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def Model_DETR(x, y):
    # Initialize the model
    input_shape = x.shape[1:]
    model = create_detr_model(input_shape, num_classes=3)

    def resize_dataset(dataset):
        resized = np.zeros((dataset.shape[0], *input_shape))
        for i in range(dataset.shape[0]):
            resized[i] = np.resize(dataset[i], input_shape)
        return resized

    # Resize the dataset if needed
    x = resize_dataset(x)
    y = resize_dataset(y)


    # Train the model
    model.fit(x, y, epochs=10, batch_size=8, validation_data=0.25)
    pred = model.predict(x, verbose=2)
    pred = np.asarray(pred)

    Eval = [net_evaluation(y[n].astype('uint8'), pred[n].astype('uint8')) for n in range(pred.shape[0])]
    return Eval, pred



