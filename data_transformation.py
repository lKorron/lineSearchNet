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