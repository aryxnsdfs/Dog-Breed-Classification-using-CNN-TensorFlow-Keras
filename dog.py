# Dog Breed Classification using Transfer Learning

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split

# Load and prepare data
labels = pd.read_csv("labels.csv")
labels["id"] = labels["id"] + ".jpg"
train_df, test_df = train_test_split(labels, test_size=0.1, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

# Image settings
IMG_SIZE = (224, 224)
BATCH = 32

# Data generators
train_gen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.2, horizontal_flip=True)
val_gen = ImageDataGenerator(rescale=1./255)

data_dir = "train/"

train_ds = train_gen.flow_from_dataframe(train_df, data_dir, x_col="id", y_col="breed",
                                         target_size=IMG_SIZE, class_mode="categorical", batch_size=BATCH)
val_ds = val_gen.flow_from_dataframe(val_df, data_dir, x_col="id", y_col="breed",
                                     target_size=IMG_SIZE, class_mode="categorical", batch_size=BATCH)

# Build model
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
out = Dense(train_ds.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=out)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ModelCheckpoint('dog_breed_best_model.h5', save_best_only=True)
]

history = model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=callbacks)

# Plot accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Training vs Validation Accuracy')
plt.show()

# Load test set for prediction
test_gen = ImageDataGenerator(rescale=1./255)
test_ds = test_gen.flow_from_dataframe(test_df, data_dir, x_col="id",
                                       class_mode=None, shuffle=False, target_size=IMG_SIZE, batch_size=BATCH)

predictions = model.predict(test_ds)
labels_map = dict((v, k) for k, v in train_ds.class_indices.items())
predicted_breeds = [labels_map[np.argmax(pred)] for pred in predictions]
