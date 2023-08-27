from scipy.optimize import line_search
from tensorflow import keras
from keras.layers import Dense, Dropout, BatchNormalization
import numpy as np
import random
import matplotlib.pyplot as plt
from keras import backend as K


def generate_coefficients(n):
    c = random.randint(-100, 100)

    def generate_a(n):
        limit = 5
        coefficients = []

        for i in range(limit):
            if i < n:
                coefficients.append(random.randint(-100, 100))
            else:
                coefficients.append(0)

        return coefficients

    return (c, generate_a(n))


def create_polynomial(c, a):
    def vector(x):
        return np.array([x, x ** 2, x ** 3, x ** 4, x ** 5])

    return lambda x: (c + np.matmul(np.array(a), vector(x[0])))


def create_gradient(c, a):
    def d_vector(x):
        return np.array([1, 2 * x, 3 * x ** 2, 4 * x ** 3, 5 * x ** 4])

    return lambda x: [np.matmul(np.array(a), d_vector(x[0]))]


def linear_search(*args):
    alpha = line_search(*args)[0]

    if alpha is None:
        return -1

    if alpha >= 1024:
        alpha = -1
    return alpha


def load_data():
    n_train = 50000
    n = 60000
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for i in range(n):
        coefficients = generate_coefficients(random.randint(1, 5))

        obj_func = create_polynomial(*coefficients)
        obj_grad = create_gradient(*coefficients)

        start_point = np.array([0, 0])
        search_gradient = np.array(obj_grad(start_point)) * -1

        alpha = linear_search(obj_func, obj_grad, start_point, search_gradient)

        if alpha == -1:
            continue

        if flattenArray(coefficients) in x_train or flattenArray(coefficients) in x_test:
            while flattenArray(coefficients) in x_train:
                coefficients = generate_coefficients(random.randint(1, 5))

        if (i > n_train):
            x_test.append(flattenArray(coefficients))
            y_test.append(alpha)
        else:
            x_train.append(flattenArray(coefficients))
            y_train.append(alpha)

    return (x_train, y_train), (x_test, y_test)


def flattenArray(array):
    result = []
    for el in array:
        if type(el) is list:
            for i in el:
                result.append(i)

        else:
            result.append(el)

    return result



(x_train, y_train), (x_test, y_test) = load_data()

x_train = np.asarray(x_train).astype(np.float32)
y_train = np.asarray(y_train).astype(np.float32)
x_test = np.asarray(x_test).astype(np.float32)
y_test = np.asarray(y_test).astype(np.float32)

# print(x_train[:10])
# print(y_train[:10])


normalizer = keras.layers.Normalization(axis=-1)
normalizer.adapt(x_train)



model = keras.Sequential([
                          normalizer,
                          # Dense(11, activation="relu"),
                          BatchNormalization(),
                          Dense(64, activation="relu"),
                          Dense(32, activation="relu"),
                          Dense(1, activation="linear")])

optimizer = keras.optimizers.Adam()
loss = keras.losses.MeanAbsoluteError()


def soft_acc(y_true, y_pred):
    return K.mean(K.equal(K.round(y_true), K.round(y_pred)))

# выбираем целевую функцию и метрику для задачи регрессии
model.compile(optimizer=optimizer, loss=loss,
              metrics=["mean_absolute_percentage_error", soft_acc])

history = model.fit(x_train, y_train, batch_size=128, epochs=160, validation_split=0.2)

results = model.evaluate(x_test, y_test, batch_size=128)

print(results)

y_predicted = model.predict(x_test[0:10])

y_real = y_test[0:10]

print(y_predicted)
print(y_real)

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])

plt.show()

# y = model.predict(flattenArray(0, [], []))




