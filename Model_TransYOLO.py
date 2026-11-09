import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

from Classificaltion_Evaluation import net_evaluation


# Define the TransYolo Model using Keras
def build_transyolo_model(input_shape=(32, 32, 3), num_classes=3):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(64, (3, 3), strides=2, padding='same', activation='relu')(inputs)
    x = layers.Conv2D(128, (3, 3), strides=2, padding='same', activation='relu')(x)
    x = layers.MultiHeadAttention(num_heads=4, key_dim=128)(x, x)

    # We now upscale the features using Conv2DTranspose (Deconvolution) to get back to (32, 32, 128)
    x = layers.Conv2DTranspose(128, (3, 3), strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(64, (3, 3), strides=2, padding='same', activation='relu')(x)

    # Final output layer (softmax for multi-class segmentation)
    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


# Define the training and evaluation function (Model_MADRL)
def Model_TransYolo(x,y):
    input_shape = (32, 32, 3)
    num_classes = 3

    # Resize all datasets to 32x32x3 (if not already)
    def resize_dataset(dataset):
        resized = np.zeros((dataset.shape[0], *input_shape))
        for i in range(dataset.shape[0]):
            resized[i] = np.resize(dataset[i], input_shape)
        return resized

    x_train = resize_dataset(x)
    y_train = resize_dataset(y)

    model = build_transyolo_model(input_shape=x_train.shape[1:], num_classes=num_classes)

    # Early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # Train the model
    model.fit(x_train, y_train, epochs=50, batch_size=16, validation_data=(x, y), callbacks=[early_stopping])

    pred = model.predict(x_train, verbose=2)
    pred = np.asarray(pred)

    Eval = [net_evaluation(y[n].astype('uint8'), pred[n].astype('uint8')) for n in range(pred.shape[0])]
    return Eval, pred

