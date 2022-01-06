import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image_dataset_from_directory
import os


PATH = os.getcwd()
IMG_SIZE = (160, 160)


# class TestModel(tf.Module):
#   def __init__(self):
#     super(TestModel, self).__init__()

#   @tf.function(input_signature=[tf.TensorSpec(shape=[1, 10], dtype=tf.float32)])
#   def add(self, x):
#     '''
#     Simple method that accepts single input 'x' and returns 'x' + 4.
#     '''
#     # Name the output 'result' for convenience.
#     return {'result' : x + 4}


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


def main():
    # saved_model_dir = "./mobilenetv2_batch32.tf"
    # converter = tf.lite.TFLiteConverter.from_saved_model(
    #     saved_model_dir, signature_keys=['serving_default'])
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # converter.experimental_new_converter = True
    # converter.target_spec.supported_ops = [
    #     tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    # tflite_model = converter.convert()

    # fo = open(
    #     "./mobilenetv2_batch32_model.tflite", "wb")
    # fo.write(tflite_model)
    # fo.close

    model = tf.keras.models.load_model('./mobilenetv2_batch32.tf')

    # model.load_weights('./mobilenetv2_batch32.tf')

    validation_dataset = get_val_dataset()
    loss, accuracy = model.evaluate(validation_dataset)
    print("Test accuracy :", accuracy)

    # model.save("./mobilenetv2_batch32.tf",save_format="tf")

    # Retrieve a batch of images from the test set
    # image_batch, label_batch = validation_dataset.as_numpy_iterator().next()
    # predictions = model.predict_on_batch(image_batch).flatten()

    # # Apply a sigmoid since our model returns logits
    # predictions = tf.nn.sigmoid(predictions)
    # predictions = tf.where(predictions < 0.5, 0, 1)

    # print("Predictions:\n", predictions.numpy())
    # print("Labels:\n", label_batch)


if __name__ == '__main__':
    main()