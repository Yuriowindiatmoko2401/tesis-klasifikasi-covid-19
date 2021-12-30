import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

from tensorflow.keras.preprocessing import image_dataset_from_directory

PATH = os.getcwd()

train_dir = os.path.join(PATH, "train")
validation_dir = os.path.join(PATH, "test")

BATCH_SIZE = 32
IMG_SIZE = (160, 160)

train_dataset = image_dataset_from_directory(
    train_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE
)

validation_dataset = image_dataset_from_directory(
    validation_dir, shuffle=True, batch_size=203, image_size=IMG_SIZE
)


class_names = train_dataset.class_names

plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")

val_batches = tf.data.experimental.cardinality(validation_dataset)
test_dataset = validation_dataset.take(val_batches // 5)
validation_dataset = validation_dataset.skip(val_batches // 5)

print(
    "Number of validation batches: %d"
    % tf.data.experimental.cardinality(validation_dataset)
)
print("Number of test batches: %d" % tf.data.experimental.cardinality(test_dataset))

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    ]
)

for image, _ in train_dataset.take(1):
    plt.figure(figsize=(10, 10))
    first_image = image[0]
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
        plt.imshow(augmented_image[0] / 255)
        plt.axis("off")

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1.0 / 127.5, offset=-1)

# Create the base model from the pre-trained model MobileNet V2
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SHAPE, include_top=False, weights="imagenet"
)

image_batch, label_batch = next(iter(train_dataset))
feature_batch = base_model(image_batch)
print(feature_batch.shape)

base_model.trainable = False
base_model.summary()

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)

prediction_layer = tf.keras.layers.Dense(1)
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)

inputs = tf.keras.Input(shape=(160, 160, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

base_learning_rate = 0.0001
model.compile(
    optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

model.summary()

len(model.trainable_variables)

initial_epochs = 25

loss0, accuracy0 = model.evaluate(validation_dataset)


print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))

history = model.fit(
    train_dataset, epochs=initial_epochs, validation_data=validation_dataset
)

acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]

loss = history.history["loss"]
val_loss = history.history["val_loss"]

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label="Training Accuracy")
plt.plot(val_acc, label="Validation Accuracy")
plt.legend(loc="lower right")
plt.ylabel("Accuracy")
plt.ylim([min(plt.ylim()), 1])
plt.title("Training and Validation Accuracy")

plt.subplot(2, 1, 2)
plt.plot(loss, label="Training Loss")
plt.plot(val_loss, label="Validation Loss")
plt.legend(loc="upper right")
plt.ylabel("Cross Entropy")
plt.ylim([0, 1.0])
plt.title("Training and Validation Loss")
plt.xlabel("epoch")
plt.show()

base_model.trainable = True

# Let's take a look to see how many layers are in the base model
print("Number of layers in the base model: ", len(base_model.layers))

# Fine-tune from this layer onwards
fine_tune_at = 100

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate / 10),
    metrics=["accuracy"],
)


model.summary()

len(model.trainable_variables)

fine_tune_epochs = 25
total_epochs = initial_epochs + fine_tune_epochs

history_fine = model.fit(
    train_dataset,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1],
    validation_data=validation_dataset,
)


acc += history_fine.history["accuracy"]
val_acc += history_fine.history["val_accuracy"]

loss += history_fine.history["loss"]
val_loss += history_fine.history["val_loss"]


plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label="Training Accuracy")
plt.plot(val_acc, label="Validation Accuracy")
plt.ylim([0.3, 1])
plt.plot(
    [initial_epochs - 1, initial_epochs - 1], plt.ylim(), label="Start Fine Tuning"
)
plt.legend(loc="lower right")
plt.title("Training and Validation Accuracy")

plt.subplot(2, 1, 2)
plt.plot(loss, label="Training Loss")
plt.plot(val_loss, label="Validation Loss")
plt.ylim([0, 1.0])
plt.plot(
    [initial_epochs - 1, initial_epochs - 1], plt.ylim(), label="Start Fine Tuning"
)
plt.legend(loc="upper right")
plt.title("Training and Validation Loss")
plt.xlabel("epoch")
plt.show()

loss, accuracy = model.evaluate(validation_dataset)
print("Test accuracy :", accuracy)

# Retrieve a batch of images from the test set
image_batch, label_batch = validation_dataset.as_numpy_iterator().next()
predictions = model.predict_on_batch(image_batch).flatten()

# Apply a sigmoid since our model returns logits
predictions = tf.nn.sigmoid(predictions)
predictions = tf.where(predictions < 0.5, 0, 1)

print("Predictions:\n", predictions.numpy())
print("Labels:\n", label_batch)

plt.figure(figsize=(10, 10))
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image_batch[i].astype("uint8"))
    plt.title(class_names[predictions[i]])
    plt.axis("off")


import sklearn
import pandas
import matplotlib
import seaborn

print("Predictions:\n", predictions.numpy())
print("Labels:\n", label_batch)

TP = 0

for i in range(0, len(label_batch)):
    if label_batch[i] == predictions.numpy()[i] and label_batch[i] == 1:
        TP += 1
print("True Positive: ", TP)

FP = 0

for i in range(0, len(label_batch)):
    if label_batch[i] == 0 and predictions.numpy()[i] == 1:
        FP += 1
print("False Positive: ", FP)

TN = 0
for i in range(0, len(label_batch)):
    if label_batch[i] == predictions.numpy()[i] and label_batch[i] == 0:
        TN += 1
print("True Negative: ", TN)

FN = 0
for i in range(0, len(label_batch)):
    if label_batch[i] == 1 and predictions.numpy()[i] == 0:
        FN += 1
print("False Negative: ", FN)

CP = 0
for i in range(0, len(label_batch)):
    if label_batch[i] == predictions.numpy()[i]:
        CP += 1
print("Correct Prediction: ", CP)
print(CP == TP + TN)

ICP = 0
for i in range(0, len(label_batch)):
    if label_batch[i] != predictions.numpy()[i]:
        ICP += 1
print("Incorrect Prediction: ", ICP)
print(ICP == FP + FN)

accuracy = (TP + TN) / (TP + FP + TN + FN)
print(accuracy * 100)

from sklearn.metrics import accuracy_score

print(accuracy_score(label_batch, predictions) * 100)

image_batch, label_batch = validation_dataset.as_numpy_iterator().next()
predictions = model.predict_on_batch(image_batch).flatten()

# Apply a sigmoid since our model returns logits
predictions = tf.nn.sigmoid(predictions)
predictions = tf.where(predictions < 0.5, 0, 1)

print(accuracy_score(label_batch, predictions.numpy()) * 100)

for i in range(0, len(label_batch)):
    if label_batch[i] == predictions.numpy()[i] and label_batch[i] == 1:
        TP += 1
for i in range(0, len(label_batch)):
    if label_batch[i] == 1 and predictions.numpy()[i] == 0:
        FN += 1
recall = (TP) / (TP + FN)
print(recall)

from sklearn.metrics import recall_score

print(recall_score(label_batch, predictions.numpy()))

image_batch, label_batch = validation_dataset.as_numpy_iterator().next()
predictions = model.predict_on_batch(image_batch).flatten()

# Apply a sigmoid since our model returns logits
predictions = tf.nn.sigmoid(predictions)
predictions = tf.where(predictions < 0.5, 0, 1)

print(accuracy_score(label_batch, predictions.numpy()) * 100)
print(recall_score(label_batch, predictions.numpy()) * 100)

TP, FP = 0, 0
for i in range(0, len(label_batch)):
    if label_batch[i] == predictions.numpy()[i] and label_batch[i] == 1:
        TP += 1
for i in range(0, len(label_batch)):
    if label_batch[i] == 0 and predictions.numpy()[i] == 1:
        FP += 1
precision = TP / (TP + FP)
print(precision)
recall = recall_score(label_batch, predictions.numpy())

from sklearn.metrics import precision_score

print(precision_score(label_batch, predictions.numpy()) * 100)

AM = (1 + 0.2) / 2
HM = 2 * (1 * 0.2) / (1 + 0.2)
print(AM)
print(HM)

f1 = 2 * (precision * recall) / (precision + recall)
print(f1)
from sklearn.metrics import f1_score

print(f1_score(label_batch, predictions.numpy()))

image_batch, label_batch = validation_dataset.as_numpy_iterator().next()
predictions = model.predict_on_batch(image_batch).flatten()

# Apply a sigmoid since our model returns logits
predictions = tf.nn.sigmoid(predictions)
predictions = tf.where(predictions < 0.5, 0, 1)

print(sum([1 for e in label_batch if e == 1]))
print(sum([1 for e in label_batch if e == 0]))

print(sum([1 for e in predictions.numpy() if e == 1]))
print(sum([1 for e in predictions.numpy() if e == 0]))

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

confusion = confusion_matrix(label_batch, predictions.numpy(), normalize="all")
FN = confusion[1][0]
TN = confusion[0][0]
TP = confusion[1][1]
FP = confusion[0][1]
plt.bar(
    ["False Negative", "True Negative", "True Positive", "False Positive"],
    [FN, TN, TP, FP],
)
plt.show()

import seaborn as sns

sns.heatmap(
    confusion,
    annot=True,
    xticklabels=["Negative", "Positive"],
    yticklabels=["Negative", "Positive"],
)
plt.ylabel("Label")
plt.xlabel("Predicted")
plt.show()

import pandas as pd

data = {"Labels": label_batch, "Predictions": predictions.numpy()}
df = pd.DataFrame(data, columns=["Labels", "Predictions"])
confusion_matrix = pd.crosstab(
    df["Labels"], df["Predictions"], rownames=["Labels"], colnames=["Predictions"]
)
print(confusion_matrix)

from sklearn.metrics import classification_report

print(classification_report(label_batch, predictions.numpy()))
