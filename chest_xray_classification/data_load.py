from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (52, 52)
CLASS_MODE = "binary"
BATCH_SIZE = 32

def load_data(train_dir, test_dir, val_dir):
    train_data = ImageDataGenerator(rescale=1/255.0)
    test_data = ImageDataGenerator(rescale=1/255.0)
    val_data = ImageDataGenerator(rescale=1/255.0)

    train_data = train_data.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        class_mode=CLASS_MODE,
        batch_size=BATCH_SIZE
    )
    test_data = test_data.flow_from_directory(
        test_dir,
        target_size=IMG_SIZE,
        class_mode=CLASS_MODE,
        batch_size=BATCH_SIZE
    )
    val_data = val_data.flow_from_directory(
        val_dir,
        target_size=IMG_SIZE,
        class_mode=CLASS_MODE,
        batch_size=BATCH_SIZE
    )

    return train_data, test_data, val_data
