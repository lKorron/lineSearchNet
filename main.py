import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from tensorflow import keras
from keras.datasets import mnist
from keras.layers import Flatten, Dense, Dropout, BatchNormalization
import numpy as np
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train)

x_train = x_train / 255
x_test = x_test / 255

y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)


model = keras.Sequential([Flatten(input_shape=(28, 28, 1)),
                          Dense(300, activation="relu"),
                          Dropout(0.8),
                          Dense(10, activation="softmax")])

model.compile(optimizer="adam", loss=keras.losses.CategoricalCrossentropy(), metrics=["accuracy"])

history = model.fit(x_train, y_train_cat, batch_size=30, epochs=50, validation_split=0.8)

# x = np.expand_dims(x_test[0], 0)
# y = model.predict(x)
# print(np.argmax(y))

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])

plt.show()