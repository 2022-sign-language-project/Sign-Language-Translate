from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import LSTM, Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

###############
# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data')

# Actions that we try to detect
actions = np.array(['default', 'foreheadhot', 'gichim', 'happy',
                   'hi', 'medicine', 'nose', 'sick', 'thanks', 'the end'])

# Thirty videos worth of data
no_sequences = 30

# Videos are going to be 30 frames in length
sequence_length = 30

# Folder start
start_folder = 30
###############

#
label_map = {label: num for num, label in enumerate(actions)}
label_map

sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(
                sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

sequences = np.array(sequences).reshape(
    no_sequences * actions.size, 30, 1662, 1)
np.array(labels).shape

X = np.array(sequences)
print(X.shape)

y = to_categorical(labels).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

print(X_test.shape)
print(y_test.shape)

#
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(30, 1662, 1)))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(48, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])

# model.add(Conv2D(input_shape=(30, 1662, 1), filters=32, kernel_size=(
#     3, 3), strides=(1, 1), padding='same'))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(
#     1, 1), padding='same'))
# # 1x1 convolution
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(
#     1, 1), padding='same'))
# # 1x1 convolution
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(
#     1, 1), padding='same'))
# # 1x1 convolution
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(
#     1, 1), padding='same'))
# # 1x1 convolution
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(
#     1, 1), padding='same'))
# # 1x1 convolution
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# # prior layer should be flattend to be connected to dense layers
# model.add(Flatten())
# # dense layer with 50 neurons
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.5))
# # final layer with 10 neurons to classify the instances
# model.add(Dense(actions.shape[0], activation='softmax'))


# model.compile(loss='categorical_crossentropy',
#               optimizer="Adam", metrics=['categorical_accuracy'])

model.fit(X_train, y_train, epochs=2000, callbacks=[tb_callback])

model.summary()


# 9. Save Weights
model.save('test_cnn_test.h5')
del model
model = load_model('test_cnn_test.h5')


# 10. Evaluation using Confusion Matrix and Accuracy
yhat = model.predict(X_test)
ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()
multilabel_confusion_matrix(ytrue, yhat)
print(accuracy_score(ytrue, yhat))
