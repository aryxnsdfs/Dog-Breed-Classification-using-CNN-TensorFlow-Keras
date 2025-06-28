import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
labels = pd.read_csv("labels.csv")
labels["id"] = labels["id"] + ".jpg"

train_df, test_df = train_test_split(labels, test_size=0.1, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)
IMG_SIZE = (224, 224)
BATCH = 32

train_gen = ImageDataGenerator(rescale=1./255,
                               rotation_range=20,
                               zoom_range=0.2,
                               horizontal_flip=True)
val_gen = ImageDataGenerator(rescale=1./255)

train_ds = train_gen.flow_from_dataframe(train_df, "train/", x_col="id",
                                         y_col="breed", target_size=IMG_SIZE,
                                         class_mode="categorical", batch_size=BATCH)
val_ds = val_gen.flow_from_dataframe(val_df, "train/", x_col="id",
                                     y_col="breed", target_size=IMG_SIZE,
                                     class_mode="categorical", batch_size=BATCH)
base = EfficientNetB0(weights="imagenet", include_top=False,
                      input_shape=IMG_SIZE + (3,))
base.trainable = False  # Freeze base

x = base.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.5)(x)
out = layers.Dense(train_ds.num_classes, activation="softmax")(x)

model = Model(inputs=base.input, outputs=out)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
es = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
mc = ModelCheckpoint("best_model.h5", save_best_only=True)

history = model.fit(train_ds,
                    validation_data=val_ds,
                    epochs=10,
                    callbacks=[es, mc])
# Plot accuracy/loss
plt.figure()
plt.plot(history.history["accuracy"], label="train_acc")
plt.plot(history.history["val_accuracy"], label="val_acc")
plt.xlabel("Epoch"), plt.ylabel("Accuracy"), plt.legend()
plt.show()
test_gen = ImageDataGenerator(rescale=1./255)
test_ds = test_gen.flow_from_dataframe(test_df, "train/", x_col="id",
                                       class_mode=None, shuffle=False,
                                       target_size=IMG_SIZE, batch_size=BATCH)

preds = model.predict(test_ds)
pred_labels = train_ds.class_indices.keys()
preds = [list(pred_labels)[np.argmax(p)] for p in preds]
