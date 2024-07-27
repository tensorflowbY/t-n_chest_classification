from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.models import Model

IMG_SIZE = (52, 52, 3)

def build_model():
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=IMG_SIZE)

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation="relu")(x)
    x = Dense(32, activation="relu")(x)
    output = Dense(1, activation="sigmoid")(x)  # For binary classification

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    print(f"Model Succes: {model.summary}")

    return model

def get_callbacks():
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
    early_stopping = EarlyStopping(monitor="val_loss", patience=5, verbose=1, min_delta=2)
    tensorboard = TensorBoard(log_dir='./logs')
    # model_checkpoint = ModelCheckpoint(filepath='best_model.weights.h5', monitor='val_loss', save_best_only=True)

    return [reduce_lr, early_stopping, tensorboard]
