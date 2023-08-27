def array_to_coefficients(array):
    N = 5
    subList = [array[n:n + N] for n in range(0, len(array), N)]
    return subList

arr = [1,2,3,4,5, 1, 2, 3, 4, 5, 100, 200, 300]



print(array_to_coefficients(arr)[:-1])


dim = 3
print(arr[-dim:])