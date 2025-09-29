# ===============================
# Hand Gesture Recognition Training (CNN+LSTM + CNN)
# ===============================

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, TimeDistributed, LSTM, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -------------------------------
# Parameters
# -------------------------------
IMG_SIZE = (64, 64)
FRAMES_PER_CLIP = 20
EPOCHS = 10
BATCH_SIZE_SEQ = 8
BATCH_SIZE_IMG = 32

# -------------------------------
# Step 1: Load gesture sequences (video-style)
# -------------------------------
def load_gesture_sequences(base_path, img_size=IMG_SIZE, frames_per_clip=FRAMES_PER_CLIP):
    X, y = [], []
    gesture_labels = sorted(os.listdir(base_path))
    label_dict = {gesture: idx for idx, gesture in enumerate(gesture_labels)}

    for gesture in gesture_labels:
        gesture_path = os.path.join(base_path, gesture)
        frame_files = sorted(os.listdir(gesture_path))

        for i in range(0, len(frame_files) - frames_per_clip + 1, frames_per_clip):
            frames = []
            for f in frame_files[i:i+frames_per_clip]:
                img_path = os.path.join(gesture_path, f)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img = cv2.resize(img, img_size)
                img = img / 255.0
                frames.append(img)
            if len(frames) == frames_per_clip:
                X.append(frames)
                y.append(label_dict[gesture])

    X = np.array(X)
    y = to_categorical(y, num_classes=len(gesture_labels))
    return X, y, label_dict

# -------------------------------
# Step 2: CNN+LSTM Model for sequences
# -------------------------------
def build_cnn_lstm_model(time_steps, height, width, channels, num_classes):
    # CNN part (no input_shape here)
    cnn = Sequential([
        Conv2D(32, (3,3), activation="relu", padding="same"),
        MaxPooling2D(2,2),
        Flatten()
    ])

    model = Sequential([
        TimeDistributed(cnn, input_shape=(time_steps, height, width, channels)),
        LSTM(64),
        Dense(128, activation="relu"),
        Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

# -------------------------------
# Step 3: CNN Model for single images
# -------------------------------
def build_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3,3), activation="relu", input_shape=input_shape, padding="same"),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation="relu", padding="same"),
        MaxPooling2D(2,2),
        Conv2D(128, (3,3), activation="relu", padding="same"),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

# -------------------------------
# Step 4: Load sequence dataset
# -------------------------------
seq_base_path = r"C:\Users\nanda\OneDrive\Documents\Nanda's ML Tasks\Hand_Gesture\archive\leapGestRecog\00"
X_seq, y_seq, label_dict = load_gesture_sequences(seq_base_path)
print("✅ Sequence data loaded:", X_seq.shape, y_seq.shape)

# Train/test split
X_train_seq, X_test_seq, y_train_seq, y_test_seq = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

# Build & train CNN+LSTM
cnn_lstm_model = build_cnn_lstm_model(
    time_steps=X_train_seq.shape[1],
    height=X_train_seq.shape[2],
    width=X_train_seq.shape[3],
    channels=X_train_seq.shape[4],
    num_classes=y_seq.shape[1]
)
cnn_lstm_model.summary()
cnn_lstm_model.fit(X_train_seq, y_train_seq, validation_data=(X_test_seq, y_test_seq), epochs=EPOCHS, batch_size=BATCH_SIZE_SEQ)
loss, acc = cnn_lstm_model.evaluate(X_test_seq, y_test_seq)
print(f"✅ CNN+LSTM Test Accuracy: {acc*100:.2f}%")
cnn_lstm_model.save("gesture_model_seq.h5")

# -------------------------------
# Step 5: Image dataset using ImageDataGenerator
# -------------------------------
train_dir = r"C:\Users\nanda\OneDrive\Documents\Nanda's ML Tasks\Hand_Gesture\dataset\train"
test_dir  = r"C:\Users\nanda\OneDrive\Documents\Nanda's ML Tasks\Hand_Gesture\dataset\test"

train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=15, zoom_range=0.1, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE_IMG, class_mode="categorical")
test_data = test_datagen.flow_from_directory(test_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE_IMG, class_mode="categorical")

cnn_model = build_cnn_model((IMG_SIZE[0], IMG_SIZE[1], 3), train_data.num_classes)
cnn_model.summary()
cnn_model.fit(train_data, validation_data=test_data, epochs=EPOCHS)
cnn_model.save("gesture_model_img.h5")

# Save labels
np.save("gesture_labels.npy", label_dict)
print("💾 All models and labels saved successfully!")
