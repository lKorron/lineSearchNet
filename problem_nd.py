import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.layers import Dense, BatchNormalization

from math_structures import create_polynomial, create_gradient, hessian, newton_method, quasi_newton_method
from data_transformation import  array_to_coefficients
from data_generation import load_data
from wolfe_conditions import is_wolfe

def train_network(dim, method="gradient", A=0, B=0):

    # загрузка данных
    (x_train, y_train), (x_test, y_test) = load_data(dim, A, B, method)

    # приведение массивов к тензорам
    x_train = np.asarray(x_train).astype(np.float32)
    y_train = np.asarray(y_train).astype(np.float32)
    x_test = np.asarray(x_test).astype(np.float32)
    y_test = np.asarray(y_test).astype(np.float32)

    # нормализация входного слоя
    normalizer = keras.layers.Normalization(axis=-1)
    normalizer.adapt(x_train)

    # структура сети
    model = keras.Sequential([
        normalizer,
        BatchNormalization(),
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(1, activation="linear")])

    # выбор оптимизатора, целевой функции
    optimizer = keras.optimizers.Adam()
    loss = keras.losses.MeanAbsoluteError()

    # def soft_acc(y_true, y_pred):
    #     return K.mean(K.equal(K.round(y_true), K.round(y_pred)))

    # компиляция, выбор метрики
    model.compile(optimizer=optimizer, loss=loss,
                  metrics=["mean_absolute_percentage_error"])

    # обучение
    history = model.fit(x_train, y_train, batch_size=128, epochs=160, validation_split=0.2)

    # тестирование
    results = model.evaluate(x_test, y_test, batch_size=128)
    print(results)

    # контроль переобучения
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.show()

    y = model.predict(x_test)
    print(y)


    def check_wolfe():
        right_values = 0

        for i in range(len(y)):
            func = create_polynomial(x_test[i][0], *array_to_coefficients(x_test[i][1:])[:-1])
            grad = create_gradient(x_test[i][0], *array_to_coefficients(x_test[i][1:])[:-1])

            point = x_test[i][-dim:]

            if method == "gradient":
                dir = np.array(grad(point)) * -1
            elif method == "newton":
                matrix = hessian(point, *array_to_coefficients(x_test[i][1:])[:-1])
                dir = newton_method(grad, matrix, point)

            elif method == "quasi_newton":
                dir = quasi_newton_method(func, grad, point)

            if is_wolfe(y[i], func, grad, point, dir):
                print(True)
                right_values += 1
            else:
                print(False)

        print("Процент соответсвия альфа усилинным условиям вольфе:")
        print(right_values / len(x_test) * 100)




    check_wolfe()

    ans = input("wanna save the model? ")

    if ans == "y":
        model.save("model1")

train_network(dim=3, method="newton", A=-10, B=10)
