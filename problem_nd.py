from scipy.optimize import line_search, minimize
from tensorflow import keras
from keras.layers import Dense, Dropout, BatchNormalization
import numpy as np
import random
import matplotlib.pyplot as plt
from keras import backend as K
from gesse import is_wolfe1, is_wolfe2, gessian, newton_method, quasi_newton_method
from numpy.linalg import det



def train_network(dim, point=None, method="gradient"):
    # генерация коэффициентов для полинома до 5й степени
    # с числом измерений, равным dim
    if point is None:
        point = np.zeros(dim)

    def generate_coefficients(n):
        c = random.randint(-100, 100)
        result = [c]

        def generate_a(n):
            limit = 5
            coefficients = []

            for i in range(limit):
                if i < n:
                    coefficients.append(random.randint(-100, 100))
                else:
                    coefficients.append(0)

            return coefficients

        for i in range(dim):
            result.append(generate_a(n))

        return result

    def create_point():
        array = []
        for i in range(dim):
            array.append(random.randint(-10, 10) / 10)

        return array

    # создание полинома через замыкание
    def create_polynomial(c, *args):
        def vector(x):
            return np.array([x, x ** 2, x ** 3, x ** 4, x ** 5])

        def polynomial(x):
            out = c
            for i in range(len(args)):
                out += np.matmul(np.array(args[i]), vector(x[i]))
            return out

        return polynomial

    # создание градиента через замыкание
    def create_gradient(c, *args):
        def d_vector(x):
            return np.array([1, 2 * x, 3 * x ** 2, 4 * x ** 3, 5 * x ** 4])

        def gradient(x):
            out = []
            for i in range(len(args)):
                out.append(np.matmul(np.array(args[i]), d_vector(x[i])))
            return out

        return gradient

    # обертка функции линейного поиска для фильтрации выборки
    def linear_search(*args):
        alpha = line_search(*args)[0]

        if alpha is None:
            return -1

        if alpha >= 1024:
            alpha = -1
        return alpha

    # функция генерации тренировочной и тестовой выборки
    def load_data():
        n_train = 50000
        n = 60000
        x_train = []
        y_train = []
        x_test = []
        y_test = []

        for i in range(n):
            coefficients = generate_coefficients(random.randint(1, 5))
            def generate_alpha():

                obj_func = create_polynomial(*coefficients)
                obj_grad = create_gradient(*coefficients)

                global start_point
                start_point = [random.randint(-5, 5) for _ in range(dim)]


                if method == "gradient":
                    dir = np.array(obj_grad(start_point)) * -1
                elif method == "newton":
                    matrix = gessian(start_point, *coefficients[1:])

                    if det(matrix) == 0:
                        return -1

                    dir = newton_method(obj_grad, matrix, start_point)

                elif method == "quasi_newton":
                    matrix = gessian(start_point, *coefficients[1:])

                    if det(matrix) == 0:
                        return -1

                    dir = quasi_newton_method(obj_func, obj_grad, start_point)


                a = linear_search(obj_func, obj_grad, start_point, dir)
                return a

            alpha = generate_alpha()

            # фильтрация выборки от ошибочных значений
            while alpha == -1:
                coefficients = generate_coefficients(random.randint(1, 5))
                alpha = generate_alpha()
            # фильтрация выборки от повторяющихся значений
            while flatten_array(coefficients) in x_train or flatten_array(coefficients) in x_test:
                coefficients = generate_coefficients(random.randint(1, 5))
                alpha = generate_alpha()

            coefficients.append(start_point)

            if (i >= n_train):
                x_test.append(flatten_array(coefficients))
                y_test.append(alpha)
            else:
                x_train.append(flatten_array(coefficients))
                y_train.append(alpha)

        return (x_train, y_train), (x_test, y_test)

    def flatten_array(array):
        result = []
        for el in array:
            if type(el) is list:
                for i in el:
                    result.append(i)

            else:
                result.append(el)

        return result

    def array_to_coefficients(array):
        N = 5
        subList = [array[n:n+N] for n in range(0, len(array), N)]
        return subList


    # загрузка данных
    (x_train, y_train), (x_test, y_test) = load_data()

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

    def soft_acc(y_true, y_pred):
        return K.mean(K.equal(K.round(y_true), K.round(y_pred)))

    # компиляция, выбор метрики
    model.compile(optimizer=optimizer, loss=loss,
                  metrics=["mean_absolute_percentage_error", soft_acc])

    # обучение
    history = model.fit(x_train, y_train, batch_size=128, epochs=160, validation_split=0.2)

    # тестирование
    results = model.evaluate(x_test, y_test, batch_size=128)
    print(results)

    # контроль переобучения
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.show()

    y = model.predict(x_train[0:50])
    print(y)


    def check_wolfe():
        right_values = 0

        for i in range(len(y)):
            func = create_polynomial(x_train[i][0], *array_to_coefficients(x_train[i][1:])[:-1])
            grad = create_gradient(x_train[i][0], *array_to_coefficients(x_train[i][1:])[:-1])

            # x_train[i][0], x_train[i][1:6], x_train[i][6:]
            point = x_train[i][-dim:]
            print(point)

            if method == "gradient":
                dir = np.array(grad(point)) * -1
            elif method == "newton":
                matrix = gessian(point, *array_to_coefficients(x_train[i][1:]))
                #x_train[i][1:6], x_train[i][6:]

                if det(matrix) == 0:
                    print("zero det")
                    return -1

                dir = newton_method(grad, matrix, point)

            elif method == "quasi_newton":
                dir = quasi_newton_method(func, grad, point)

            if is_wolfe1(y[i], func, grad, point, dir) and is_wolfe2(y[i], func, grad, point, dir):
                print(True)
                right_values += 1
            else:
                print(False)

        print(right_values / 50)




    check_wolfe()

train_network(dim=3)
