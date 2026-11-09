import numpy as np
import keras
import tensorflow as tf
from tensorflow.keras import layers, activations
from tensorflow.keras import Model
from keras.optimizers import Adam

from Classificaltion_Evaluation import net_evaluation


# Helper function for MLP layers
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=activations.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

# Patches layer for Vision Transformer
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config

# PatchEncoder layer for Vision Transformer
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)

    def call(self, patches):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patches) + self.position_embedding(positions)
        return encoded

    def get_config(self):
        config = super().get_config()
        config.update({"num_patches": self.num_patches})
        return config

# Deformable Attention Layer (for Deformable DETR)
class DeformableAttention(layers.Layer):
    def __init__(self, num_heads, key_dim, num_levels=4, num_points=4, dropout=0.1):
        super(DeformableAttention, self).__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.num_levels = num_levels
        self.num_points = num_points
        self.dropout = dropout
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=key_dim, dropout=dropout
        )

    def call(self, query, value):
        # Simulate deformable attention by adjusting the attention
        # mechanism to focus on a sparse set of key locations (num_points).
        return self.attention(query, value, value)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "key_dim": self.key_dim,
            "num_levels": self.num_levels,
            "num_points": self.num_points,
            "dropout": self.dropout
        })
        return config

#  Deformable DETR-based model
def create_deformable_detr_model(input_shape, num_patches, projection_dim, transformer_layers, transformer_units,
                                 num_heads, mlp_head_units, num_classes):
    inputs = keras.Input(shape=input_shape)
    augmented = layers.Normalization()(inputs)

    # Create patches and encode them with ViT
    patches = Patches(patch_size=6)(augmented)
    encoded_patches = PatchEncoder(num_patches=num_patches, projection_dim=projection_dim)(patches)

    # Apply deformable attention layers (instead of traditional attention)
    for _ in range(transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        deformable_attention_output = DeformableAttention(
            num_heads=num_heads, key_dim=projection_dim
        )(x1, x1)
        x2 = layers.Add()([deformable_attention_output, encoded_patches])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        encoded_patches = layers.Add()([x3, x2])

    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    output = layers.Dense(32 * 32 * 3, activation='softmax')(features)
    output = layers.Reshape((32, 32, 3))(output)

    model = keras.Model(inputs=inputs, outputs=output)
    return model

def Model_ViT_Deformable_DETR(train_data, train_target):
    BS = 4
    IMG_SIZE = 32
    num_classes = 1

    # Resize the input data
    Train_X = np.zeros((train_data.shape[0], IMG_SIZE, IMG_SIZE, 3))
    for i in range(train_data.shape[0]):
        temp = np.resize(train_data[i], (IMG_SIZE, IMG_SIZE, 3))
        Train_X[i] = temp
    input_shape = (IMG_SIZE, IMG_SIZE, 3)
    num_patches = (IMG_SIZE // 6) ** 2
    projection_dim = 64
    transformer_layers = 8
    transformer_units = [projection_dim * 2, projection_dim]
    num_heads = 4
    mlp_head_units = [2048, 1024]

    # Create and compile the Deformable DETR model
    vit_deformable_detr_model = create_deformable_detr_model(
        input_shape, num_patches, projection_dim, transformer_layers,
        transformer_units, num_heads, mlp_head_units, num_classes
    )
    vit_deformable_detr_model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    vit_deformable_detr_model.summary()

    # Train the model
    vit_deformable_detr_model.fit(Train_X, train_target, batch_size=BS, epochs=10, steps_per_epoch=10,
                                  validation_data=0.25)

    # Evaluate the model using the trained model
    pred = Model.predict(train_data, verbose=2)
    pred = np.asarray(pred)

    Eval = [net_evaluation(train_target[n].astype('uint8'), pred[n].astype('uint8')) for n in range(pred.shape[0])]
    return Eval, pred




