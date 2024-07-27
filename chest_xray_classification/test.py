from tensorflow.keras.models import load_model
import data_load

TRAIN_DIR = ""
TEST_DIR = ""
VAL_DIR = ""

train_data, test_data, val_data = data_load.load_data(TRAIN_DIR, TEST_DIR, VAL_DIR)

model = load_model("model.keras")

predict=model.evaluate(test_data)

print(predict)