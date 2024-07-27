import data_processing
import data_load
import model_vgg

TRAIN_DIR = ""
TEST_DIR = ""
VAL_DIR = ""
EPOCHS = 15

def main():

    train_data, test_data, val_data = data_load.load_data(TRAIN_DIR, TEST_DIR, VAL_DIR)


    data_processing.preprocessing(TRAIN_DIR)
    

    model_instance = model_vgg.build_model()


    callbacks = model_vgg.get_callbacks()

    # TRAİN MODEL
    model_instance.fit(
        train_data,
        epochs=EPOCHS,
        validation_data=val_data,
        callbacks=callbacks
    )

    model_instance.save("model.keras")

if __name__ == "__main__":
    main()
