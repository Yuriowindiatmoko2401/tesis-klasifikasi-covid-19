import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image_dataset_from_directory
import os, time

import matplotlib.pyplot as plt

tf.config.list_physical_devices("GPU")

PATH = os.getcwd()
IMG_SIZE = (160, 160)


def get_val_dataset():
    validation_dir = os.path.join(PATH, "./test")
    validation_dataset = image_dataset_from_directory(
        validation_dir, shuffle=True, batch_size=203, image_size=IMG_SIZE
    )
    val_batches = tf.data.experimental.cardinality(validation_dataset)
    test_dataset = validation_dataset.take(val_batches // 5)
    validation_dataset = validation_dataset.skip(val_batches // 5)

    print(
        "Number of validation batches: %d"
        % tf.data.experimental.cardinality(validation_dataset)
    )
    AUTOTUNE = tf.data.AUTOTUNE
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
    return validation_dataset


def predict_by_one(input_img_fl32):
    predictions = model.predict(input_img_fl32[np.newaxis, :, :, :])
    predictions = tf.nn.sigmoid(predictions)
    predictions = tf.where(predictions < 0.5, 0, 1)
    return predictions.numpy()


def predict_by_one_tflite(input_img_fl32):
    #     input_shape = input_details[0]["shape"]
    interpreter.set_tensor(
        input_details[0]["index"], input_img_fl32[np.newaxis, :, :, :]
    )
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]["index"])
    predictions = tf.nn.sigmoid(output_data)
    predictions = tf.where(predictions < 0.5, 0, 1)
    return predictions.numpy()


model = tf.keras.models.load_model("./mobilenetv2_batch32.tf")
validation_dataset_prefetch = get_val_dataset()
loss, accuracy = model.evaluate(validation_dataset_prefetch)
print("Test accuracy :", accuracy)

validation_dir = os.path.join(PATH, "./test")
validation_dataset = image_dataset_from_directory(
    validation_dir, shuffle=True, batch_size=203, image_size=IMG_SIZE
)
image_batch, label_batch = validation_dataset.as_numpy_iterator().next()

print(label_batch[1])
plt.imshow(image_batch[1].astype(np.uint8))
plt.show()

# performance model non-tflite
count = 0
for i in range(len(label_batch)):
    before = time.time()
    prediction = predict_by_one(image_batch[i])[0][0]
    in_ms_inf = time.time() - before
    print(in_ms_inf)
    if prediction == label_batch[i]:
        count += 1
        print(count)
#     print(predict_by_one(image_batch[i])[0][0])
#     print(label_batch[i])

interpreter = tf.lite.Interpreter(model_path="./mobilenetv2_batch32.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# performance model tflite
count = 0
for i in range(len(label_batch)):
    before = time.time()
    prediction = predict_by_one_tflite(image_batch[i])[0][0]
    in_ms_inf = time.time() - before
    print(in_ms_inf)
    if prediction == label_batch[i]:
        count += 1
        print(count)
#     print(predict_by_one(image_batch[i])[0][0])
#     print(label_batch[i])
